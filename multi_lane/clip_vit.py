from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_lane.blocks import Block, TaskIdentifier


class ClipViTB16Patch(nn.Module):
    """Frozen CLIP ViT-B/16 tokens with the original MULTI-LANE task pathway."""

    def __init__(
            self, num_classes=1000, global_pool='token', drop_rate=0.,
            drop_path_rate=0., norm_layer=None, act_layer=None, args=None,
            **kwargs):
        super().__init__()
        assert global_pool in ('', 'token')

        import open_clip

        self.num_classes = num_classes
        self.global_pool = global_pool
        clip_name = getattr(args, 'clip_model_name', 'ViT-B-16')
        clip_pretrained = getattr(args, 'clip_pretrained', 'laion400m_e32')
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_name, pretrained=clip_pretrained)
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.visual = self.clip_model.visual
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        self.num_features = self.embed_dim = self.visual.conv1.weight.shape[0]
        self.clip_text_dim = self._infer_clip_text_dim()
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False
        self.head_mode = getattr(args, 'head_mode', 'concat')
        self.debug_shapes = bool(getattr(args, 'clip_debug_shapes', False))
        self._debug_shapes_printed = False
        self.train_logit_scale = bool(getattr(args, 'train_logit_scale', False))
        self.text_templates = self._parse_text_templates(args)
        self.class_names = None
        self._text_feature_cache_key = None
        self._text_feature_class_ids = tuple()
        self._last_clip_text_debug_shapes = None
        self._last_ddp_debug_shapes = None
        self._debug_text_cache_keys_printed = set()
        self._ddp_text_token_cache = {}
        self.register_buffer('text_features', torch.empty(0), persistent=False)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        clip_blocks = self.visual.transformer.resblocks
        act_layer = act_layer or type(clip_blocks[0].mlp.gelu)
        depth = len(clip_blocks)
        num_heads = clip_blocks[0].attn.num_heads
        mlp_ratio = 4.
        qkv_bias = True
        ddp_head = self._is_ddp_head()
        pret_attention = False if ddp_head else (True if args is None else args.method in ['prompts'])
        num_prompt_layers = 5 if args is None else args.num_prompt_layers
        prompts = [True] * num_prompt_layers + [False] * max(0, depth - num_prompt_layers)
        prompts = prompts[:depth]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, pret_attention=pret_attention, prompts=prompts[i], id=i)
            for i in range(depth)
        ])
        self.norm = self.visual.ln_post
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.visual_proj = nn.Linear(self.embed_dim, self.clip_text_dim, bias=False)
        self.text_logit_bias = nn.Parameter(torch.zeros(num_classes))

        self._load_clip_visual_transformer_weights()
        self.head.apply(self._init_weights)
        self._init_visual_projection()
        if ddp_head:
            self._init_ddp_prompts(args, depth)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _infer_clip_text_dim(self):
        if hasattr(self.visual, 'proj') and self.visual.proj is not None:
            return self.visual.proj.shape[-1]
        if hasattr(self.clip_model, 'text_projection'):
            return self.clip_model.text_projection.shape[-1]
        return 512

    def _parse_text_templates(self, args):
        templates = getattr(args, 'text_templates', None)
        if templates is None:
            templates = [getattr(args, 'text_template', 'a photo of a {}.')]
        elif isinstance(templates, str):
            templates = [templates]
        return [template for template in templates if template]

    def _init_visual_projection(self):
        clip_proj = getattr(self.visual, 'proj', None)
        if clip_proj is not None and tuple(clip_proj.shape) == (self.embed_dim, self.clip_text_dim):
            with torch.no_grad():
                self.visual_proj.weight.copy_(clip_proj.detach().float().t())
            return
        self._init_weights(self.visual_proj)

    def _is_ddp_head(self):
        return self.head_mode in ('clip_ddp', 'ddp')

    def _init_ddp_prompts(self, args, depth):
        self.ddp_prompt_length = int(getattr(args, 'ddp_prompt_length', 16))
        self.ddp_prompt_layers = int(getattr(args, 'ddp_prompt_layers', 5))
        self.ddp_prompt_layers = max(0, min(self.ddp_prompt_layers, depth))
        self.ddp_visual_start_layer = depth - self.ddp_prompt_layers
        self.ddp_tau_max = float(getattr(args, 'ddp_tau_max', 3.0))
        self.ddp_gamma = float(getattr(args, 'ddp_gamma', 0.7))
        self.ddp_pcd = bool(getattr(args, 'ddp_pcd', True))
        self.ddp_similarity_aggregation = getattr(args, 'ddp_similarity_aggregation', 'mean')
        self.ddp_class_chunk_size = max(1, int(getattr(args, 'ddp_class_chunk_size', 4)))
        self.text_width = self.clip_model.token_embedding.weight.shape[-1]
        self.context_length = self.clip_model.positional_embedding.shape[0]

        text_prompts = torch.empty(self.num_classes, 2, self.ddp_prompt_length, self.text_width)
        visual_prompts = torch.empty(self.num_classes, 2, self.ddp_prompt_length, self.embed_dim)
        nn.init.normal_(text_prompts, std=0.02)
        nn.init.normal_(visual_prompts, std=0.02)
        self.ddp_text_prompts = nn.Parameter(text_prompts)
        self.ddp_visual_prompts = nn.Parameter(visual_prompts)
        self.register_buffer('ddp_trainable_class_mask', torch.zeros(self.num_classes), persistent=False)
        self.ddp_text_prompts.register_hook(self._mask_ddp_prompt_grad)
        self.ddp_visual_prompts.register_hook(self._mask_ddp_prompt_grad)

    def _mask_ddp_prompt_grad(self, grad):
        if grad is None:
            return None
        mask = self.ddp_trainable_class_mask.to(device=grad.device, dtype=grad.dtype)
        return grad * mask.view(-1, 1, 1, 1)

    def _set_ddp_trainable_classes(self, class_ids):
        mask = torch.zeros_like(self.ddp_trainable_class_mask)
        if len(class_ids) > 0:
            class_ids = torch.as_tensor(class_ids, device=mask.device, dtype=torch.long)
            mask.index_fill_(0, class_ids, 1.)
        self.ddp_trainable_class_mask.copy_(mask)

    def _refresh_ddp_trainable_mask(self):
        if not self._is_ddp_head():
            return
        if hasattr(self, 'class_mask') and self.class_mask is not None:
            class_ids = self.class_mask[int(self.t)]
        else:
            class_ids = list(range(self.num_classes))
        self._set_ddp_trainable_classes(class_ids)

    def _ddp_temperature(self, task_ids, device, dtype, eval):
        if not eval or not self.ddp_pcd:
            return torch.ones((), device=device, dtype=dtype)
        if not hasattr(self, 'class_mask') or self.class_mask is None or len(self.class_mask) == 0:
            return torch.ones((), device=device, dtype=dtype)

        base_count = len(self.class_mask[0])
        seen_count = sum(len(self.class_mask[int(task_id)]) for task_id in task_ids)
        total_count = self.num_classes
        denom = max(1, total_count - base_count)
        progress = (seen_count - base_count) / denom
        progress = min(1.0, max(0.0, progress))
        tau = 1.0 + (self.ddp_tau_max - 1.0) * (progress ** self.ddp_gamma)
        return torch.as_tensor(tau, device=device, dtype=dtype)

    @property
    def clip_dtype(self):
        return self.visual.conv1.weight.dtype

    def _load_clip_visual_transformer_weights(self):
        for block, clip_block in zip(self.blocks, self.visual.transformer.resblocks):
            block.norm1.load_state_dict(clip_block.ln_1.state_dict())
            block.norm2.load_state_dict(clip_block.ln_2.state_dict())
            block.attn.qkv.weight.data.copy_(clip_block.attn.in_proj_weight.data)
            block.attn.qkv.bias.data.copy_(clip_block.attn.in_proj_bias.data)
            block.attn.proj.weight.data.copy_(clip_block.attn.out_proj.weight.data)
            block.attn.proj.bias.data.copy_(clip_block.attn.out_proj.bias.data)
            block.mlp.fc1.weight.data.copy_(clip_block.mlp.c_fc.weight.data)
            block.mlp.fc1.bias.data.copy_(clip_block.mlp.c_fc.bias.data)
            block.mlp.fc2.weight.data.copy_(clip_block.mlp.c_proj.weight.data)
            block.mlp.fc2.bias.data.copy_(clip_block.mlp.c_proj.bias.data)

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.text_logit_bias = nn.Parameter(torch.zeros(num_classes))
        self._ddp_text_token_cache = {}

    def set_class_names(self, class_names):
        self.class_names = [str(class_name) for class_name in class_names]
        self._text_feature_cache_key = None

    def _get_class_names(self):
        if self.class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
        else:
            class_names = list(self.class_names)

        if len(class_names) < self.num_classes:
            class_names.extend(str(i) for i in range(len(class_names), self.num_classes))
        return class_names[:self.num_classes]

    def _task_class_ids(self, task_id, device):
        if hasattr(self, 'class_mask') and self.class_mask is not None:
            class_ids = self.class_mask[int(task_id)]
        else:
            class_ids = list(range(self.num_classes))
        return torch.as_tensor(class_ids, device=device, dtype=torch.long)

    def _seen_class_ids(self, task_ids, device):
        if hasattr(self, 'class_mask') and self.class_mask is not None:
            class_ids = []
            for task_id in task_ids:
                class_ids.extend(int(class_id) for class_id in self.class_mask[int(task_id)])
        else:
            class_ids = list(range(self.num_classes))
        return torch.as_tensor(class_ids, device=device, dtype=torch.long)

    @torch.no_grad()
    def build_text_features(self, class_names, device, dtype):
        text_features = []
        self.clip_model.eval()
        for class_name in class_names:
            prompts = [template.format(class_name) for template in self.text_templates]
            text_tokens = self.tokenizer(prompts).to(device)
            class_features = self.clip_model.encode_text(text_tokens).float()
            class_features = F.normalize(class_features, dim=-1)
            class_feature = F.normalize(class_features.mean(dim=0), dim=0)
            text_features.append(class_feature)
        return torch.stack(text_features, dim=0).to(device=device, dtype=dtype).detach()

    def update_text_features(self, seen_class_ids, device, dtype):
        class_ids = tuple(int(class_id) for class_id in seen_class_ids.detach().cpu().tolist())
        class_names = self._get_class_names()
        selected_names = tuple(class_names[class_id] for class_id in class_ids)
        cache_key = (class_ids, selected_names, tuple(self.text_templates), device.type, device.index, str(dtype))

        if self._text_feature_cache_key != cache_key:
            self._debug_print_text_prompts(class_ids, selected_names)
            self.text_features = self.build_text_features(selected_names, device=device, dtype=dtype)
            self._text_feature_cache_key = cache_key
            self._text_feature_class_ids = class_ids
        return self.text_features

    def _debug_print_text_prompts(self, class_ids, class_names):
        if not self.debug_shapes:
            return

        debug_key = (class_ids, class_names, tuple(self.text_templates))
        if debug_key in self._debug_text_cache_keys_printed:
            return
        self._debug_text_cache_keys_printed.add(debug_key)

        preview_count = min(5, len(class_names))
        preview = []
        for class_id, class_name in zip(class_ids[:preview_count], class_names[:preview_count]):
            prompts = [template.format(class_name) for template in self.text_templates]
            preview.append({
                'class_id': class_id,
                'class_name': class_name,
                'prompts': prompts,
            })

        print('[clip_vit_b16_patch debug] text class ids:', class_ids)
        print('[clip_vit_b16_patch debug] text class names preview:', tuple(class_names[:preview_count]))
        for item in preview:
            print(
                f"[clip_vit_b16_patch debug] prompts for class "
                f"{item['class_id']} ({item['class_name']}): {item['prompts']}"
            )

    def _text_feature_rows(self, class_ids, device):
        row_by_class_id = {class_id: row for row, class_id in enumerate(self._text_feature_class_ids)}
        rows = [row_by_class_id[int(class_id)] for class_id in class_ids.detach().cpu().tolist()]
        return torch.as_tensor(rows, device=device, dtype=torch.long)

    def _ddp_class_text_tokens(self, class_id, class_name):
        cache_key = (int(class_id), class_name, self.ddp_prompt_length, self.context_length)
        cached = self._ddp_text_token_cache.get(cache_key)
        if cached is not None:
            return cached

        empty_tokens = self.tokenizer([''])[0]
        sot_token = int(empty_tokens[0].item())
        eot_token = int(empty_tokens.max().item())
        class_tokens = self.tokenizer([f'{class_name}.'])[0]
        eot_pos = int(class_tokens.argmax().item())
        class_tokens = class_tokens[1:eot_pos]

        token_ids = torch.zeros(self.context_length, dtype=torch.long)
        token_ids[0] = sot_token
        class_start = 1 + self.ddp_prompt_length
        max_class_tokens = max(0, self.context_length - class_start - 1)
        class_tokens = class_tokens[:max_class_tokens]
        if class_tokens.numel() > 0:
            token_ids[class_start:class_start + class_tokens.numel()] = class_tokens
        token_ids[class_start + class_tokens.numel()] = eot_token
        self._ddp_text_token_cache[cache_key] = token_ids
        return token_ids

    def _encode_ddp_text_embeddings(self, token_embeddings, token_ids):
        dtype = self.clip_model.token_embedding.weight.dtype
        x = token_embeddings.to(dtype=dtype)
        x = x + self.clip_model.positional_embedding.to(device=x.device, dtype=dtype)
        x = x.permute(1, 0, 2)
        try:
            x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        except TypeError:
            x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).float()
        eot_indices = token_ids.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices]
        text_projection = getattr(self.clip_model, 'text_projection', None)
        if text_projection is not None:
            x = x @ text_projection.float()
        return x

    def _ddp_text_features(self, class_ids, device, dtype):
        class_names = self._get_class_names()
        token_rows = []
        for class_id in class_ids.detach().cpu().tolist():
            token_ids = self._ddp_class_text_tokens(class_id, class_names[int(class_id)])
            token_rows.extend([token_ids, token_ids.clone()])

        token_ids = torch.stack(token_rows, dim=0).to(device=device)
        token_embeddings = self.clip_model.token_embedding(token_ids)
        prompts = self.ddp_text_prompts.index_select(0, class_ids)
        prompts = prompts.reshape(class_ids.numel() * 2, self.ddp_prompt_length, self.text_width)
        token_embeddings[:, 1:1 + self.ddp_prompt_length] = prompts.to(
            device=device, dtype=token_embeddings.dtype)
        text_features = self._encode_ddp_text_embeddings(token_embeddings, token_ids)
        text_features = text_features.reshape(class_ids.numel(), 2, self.clip_text_dim)
        return F.normalize(text_features.to(device=device, dtype=dtype), dim=-1)

    def _forward_ddp_prompt_attention(self, attention, x, prompts):
        prompt_len = prompts.size(1)
        prompted = torch.cat((prompts, x), dim=1)
        batch_size, token_count, dim = prompted.shape
        qkv = attention.qkv(prompted)
        qkv = qkv.reshape(batch_size, token_count, 3, attention.num_heads, dim // attention.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * attention.scale
        attn = attn.softmax(dim=-1)
        attn = attention.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, dim)
        x = attention.proj(x)
        x = attention.proj_drop(x)
        return x[:, prompt_len:]

    def _forward_ddp_prompt_block(self, block, x, prompts):
        attn_out = self._forward_ddp_prompt_attention(block.attn, block.norm1(x), prompts)
        x = x + block.drop_path1(block.ls1(attn_out))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def _ddp_visual_features_for_chunk(self, base_tokens, class_ids):
        batch_size = base_tokens.size(0)
        pair_count = class_ids.numel() * 2
        tokens = base_tokens.unsqueeze(0).expand(pair_count, -1, -1, -1)
        tokens = tokens.reshape(pair_count * batch_size, base_tokens.size(1), base_tokens.size(2))
        prompts = self.ddp_visual_prompts.index_select(0, class_ids)
        prompts = prompts.reshape(pair_count, self.ddp_prompt_length, self.embed_dim)
        prompts = prompts.repeat_interleave(batch_size, dim=0).to(device=tokens.device, dtype=tokens.dtype)

        for block in self.blocks[self.ddp_visual_start_layer:]:
            tokens = self._forward_ddp_prompt_block(block, tokens, prompts)

        tokens = self.norm(tokens)
        tokens = self.visual_proj(tokens)
        tokens = F.normalize(tokens, dim=-1)
        return tokens.reshape(class_ids.numel(), 2, batch_size, tokens.size(1), self.clip_text_dim)

    def _aggregate_ddp_similarity(self, visual_tokens, text_features):
        text_features = text_features[:, :, None, None, :]
        similarities = (visual_tokens * text_features).sum(dim=-1)
        if self.ddp_similarity_aggregation == 'max':
            similarities = similarities.max(dim=-1).values
        elif self.ddp_similarity_aggregation == 'cls':
            similarities = similarities[:, :, :, 0]
        else:
            similarities = similarities.mean(dim=-1)
        return similarities.permute(2, 0, 1)

    def _forward_ddp_head(self, x, eval=False):
        self.clip_model.eval()
        self._refresh_ddp_trainable_mask()
        device = x.device
        batch_size = x.size(0)

        with torch.no_grad():
            base_tokens = self.extract_clip_stem_tokens(x)
            for block in self.blocks[:self.ddp_visual_start_layer]:
                base_tokens = block(base_tokens)
        base_tokens = base_tokens.detach()
        dtype = base_tokens.dtype

        if self.training and not eval:
            task_ids = [self.t]
        else:
            task_ids = list(range(self.t + 1))
        class_ids = self._seen_class_ids(task_ids, device=device)
        tau = self._ddp_temperature(task_ids, device=device, dtype=dtype, eval=eval)
        logit_scale = self.clip_model.logit_scale.exp().clamp(max=100).to(device=device, dtype=dtype)
        final_logits = base_tokens.new_zeros((batch_size, self.num_classes))
        first_scores = None

        for start in range(0, class_ids.numel(), self.ddp_class_chunk_size):
            chunk_class_ids = class_ids[start:start + self.ddp_class_chunk_size]
            text_features = self._ddp_text_features(chunk_class_ids, device=device, dtype=dtype)
            visual_features = self._ddp_visual_features_for_chunk(base_tokens, chunk_class_ids)
            scores = self._aggregate_ddp_similarity(visual_features, text_features)
            logits = logit_scale * (scores[:, :, 0] - scores[:, :, 1]) / tau
            final_logits[:, chunk_class_ids] = logits
            if first_scores is None:
                first_scores = scores

        self._last_ddp_debug_shapes = {
            'base_tokens': tuple(base_tokens.shape),
            'class_ids': tuple(class_ids.shape),
            'ddp_text_prompts': tuple(self.ddp_text_prompts.shape),
            'ddp_visual_prompts': tuple(self.ddp_visual_prompts.shape),
            'first_score_pair': tuple(first_scores.shape) if first_scores is not None else None,
            'tau': float(tau.detach().cpu().item()),
            'logit_scale': float(logit_scale.detach().cpu().item()),
            'final_logits': tuple(final_logits.shape),
        }
        sim = torch.ones((), device=device, dtype=dtype)
        tasks = torch.zeros(batch_size, dtype=torch.long, device=device)
        return final_logits, base_tokens, base_tokens[:, 0], sim, tasks, base_tokens[:, 0]

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def extract_clip_stem_tokens(self, x):
        self.clip_model.eval()
        with torch.no_grad():
            x = x.to(dtype=self.clip_dtype)
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            patch_tokens = x.permute(0, 2, 1)

            cls_token = self.visual.class_embedding.to(patch_tokens.dtype)
            cls_token = cls_token + torch.zeros(
                patch_tokens.shape[0], 1, patch_tokens.shape[-1],
                dtype=patch_tokens.dtype, device=patch_tokens.device)
            tokens = torch.cat([cls_token, patch_tokens], dim=1)
            tokens = tokens + self.visual.positional_embedding.to(tokens.dtype)
            tokens = self.visual.ln_pre(tokens)
        return tokens.float()

    def _get_selectors(self):
        assert hasattr(self, 'selectors')
        if self.training:
            return self.selectors[self.t:self.t + 1]
        return self.selectors[:self.t + 1]

    def forward_features(self, x):
        token_features = self.extract_clip_stem_tokens(x)
        stem_cls_token = token_features[:, 0:1]

        if hasattr(self, 'selectors'):
            task_tokens = self._get_selectors()
        else:
            task_num = 1 if self.training else self.t + 1
            task_tokens = torch.zeros(
                (task_num, self.num_selectors, self.embed_dim),
                device=token_features.device,
                dtype=token_features.dtype,
            )

        task_tokens = task_tokens.unsqueeze(1).expand(-1, token_features.size(0), -1, -1)
        cls_for_tasks = stem_cls_token.unsqueeze(0).expand(task_tokens.size(0), -1, -1, -1)
        task_tokens = torch.cat((cls_for_tasks, task_tokens), dim=2)

        for block in self.blocks:
            token_features, task_tokens = block(token_features, task_tokens=task_tokens)

        token_features = self.norm(token_features)
        task_tokens = self.norm(task_tokens)
        cls_token = token_features[:, 0:1]
        patch_tokens = token_features[:, 1:]
        return token_features, task_tokens, patch_tokens, cls_token

    def _debug_print_shapes(
            self, input_image, patch_tokens, cls_token, summarized_tokens,
            task_tokens, classifier_input, logits):
        if not self.debug_shapes or self._debug_shapes_printed:
            return
        print('[clip_vit_b16_patch debug] input image:', tuple(input_image.shape))
        print('[clip_vit_b16_patch debug] clip patch tokens:', tuple(patch_tokens.shape))
        print('[clip_vit_b16_patch debug] clip cls token:', tuple(cls_token.shape))
        print('[clip_vit_b16_patch debug] summarized tokens:', tuple(summarized_tokens.shape))
        print('[clip_vit_b16_patch debug] task tokens:', tuple(task_tokens.shape))
        print('[clip_vit_b16_patch debug] classifier input:', tuple(classifier_input.shape))
        print('[clip_vit_b16_patch debug] logits:', tuple(logits.shape))
        if self._last_clip_text_debug_shapes is not None:
            for name, shape in self._last_clip_text_debug_shapes.items():
                print(f'[clip_vit_b16_patch debug] {name}:', shape)
        if self._last_ddp_debug_shapes is not None:
            for name, shape in self._last_ddp_debug_shapes.items():
                print(f'[clip_vit_b16_patch debug] ddp {name}:', shape)
        self._debug_shapes_printed = True

    def _forward_clip_task_text_head(self, token_features, task_tokens, eval):
        cls_feature = token_features[:, 0]
        device = cls_feature.device
        batch_size = cls_feature.size(0)
        dtype = cls_feature.dtype
        num_task_tokens, _, token_count, token_dim = task_tokens.size()
        assert token_dim == self.embed_dim
        assert token_count >= 1

        if self.training and not eval:
            task_ids = [self.t]
            seen_task_ids = list(range(self.t + 1))
        else:
            task_ids = list(range(num_task_tokens))
            seen_task_ids = task_ids

        seen_class_ids = self._seen_class_ids(seen_task_ids, device=device)
        text_features = self.update_text_features(seen_class_ids, device=device, dtype=dtype)
        final_logits = task_tokens.new_zeros((batch_size, self.num_classes))
        summarized_tokens = task_tokens.permute(1, 0, 2, 3).clone()
        logit_scale = self.clip_model.logit_scale.exp().clamp(max=100).to(device=device, dtype=dtype)

        debug_shapes = {
            'task_output_tokens': tuple(task_tokens.shape),
            'text_features': tuple(text_features.shape),
        }
        first_projected_task_cls = None

        for local_task_idx, task_id in enumerate(task_ids):
            task_cls = task_tokens[local_task_idx, :, 0, :]
            assert task_cls.shape == (batch_size, self.embed_dim)

            projected_task_cls = self.visual_proj(task_cls)
            projected_task_cls = F.normalize(projected_task_cls, dim=-1)
            assert projected_task_cls.shape == (batch_size, self.clip_text_dim)

            task_class_ids = self._task_class_ids(task_id, device=device)
            text_rows = self._text_feature_rows(task_class_ids, device=device)
            task_text_features = text_features.index_select(0, text_rows)
            task_text_features = F.normalize(task_text_features, dim=-1)
            assert task_text_features.shape == (task_class_ids.numel(), self.clip_text_dim)

            similarity = projected_task_cls @ task_text_features.t()
            assert similarity.shape == (batch_size, task_class_ids.numel())
            task_bias = self.text_logit_bias.index_select(0, task_class_ids).to(device=device, dtype=dtype)
            logits_task = logit_scale * similarity + task_bias
            assert logits_task.shape == (batch_size, task_class_ids.numel())
            final_logits[:, task_class_ids] = logits_task

            if local_task_idx == 0:
                first_projected_task_cls = projected_task_cls
                debug_shapes.update({
                    'task_cls': tuple(task_cls.shape),
                    'projected_task_cls': tuple(projected_task_cls.shape),
                    'task_text_features': tuple(task_text_features.shape),
                    'similarity': tuple(similarity.shape),
                    'task_bias': tuple(task_bias.shape),
                    'logits_task': tuple(logits_task.shape),
                })

        debug_shapes['final_logits'] = tuple(final_logits.shape)
        self._last_clip_text_debug_shapes = debug_shapes
        sim = torch.ones((), device=device, dtype=dtype)
        tasks = torch.zeros(batch_size, dtype=torch.long, device=device)
        classifier_input = first_projected_task_cls
        return final_logits, summarized_tokens, cls_feature, sim, tasks, classifier_input

    def forward_head(self, token_features, task_tokens, eval: bool = False):
        cls_feature = token_features[:, 0]
        device = cls_feature.device
        batch_size = cls_feature.size(0)
        sim = torch.ones((), device=device, dtype=cls_feature.dtype)
        tasks = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._last_clip_text_debug_shapes = None

        if self.head_mode in ('clip_taskCLS_text', 'clip_task_cls_text'):
            return self._forward_clip_task_text_head(token_features, task_tokens, eval)

        if self.head_mode == 'task' or not eval:
            if hasattr(self, 'task_identifier'):
                tasks, sim = self.task_identifier(cls_feature.detach())
                tasks = tasks.to(device=device, dtype=torch.long)
                sim = sim.to(device=device, dtype=cls_feature.dtype)

            summarized_tokens = task_tokens.permute(1, 0, 2, 3).clone()
            task_cls = task_tokens[:, :, 0].permute(1, 0, 2)
            classifier_input = task_cls[torch.arange(batch_size, device=device), tasks]
            if self.normalize == 'pre-head':
                classifier_input = F.normalize(classifier_input, dim=-1)

            logits = self.head(classifier_input)
            return logits, summarized_tokens, cls_feature, sim, tasks, classifier_input

        if self.head_mode in ('concat', 'learned_classifier'):
            num_tasks, batch_size, _, _ = task_tokens.size()
            classifier_input = task_tokens[:, :, 0].permute(1, 0, 2)
            if self.normalize == 'pre-head':
                classifier_input = F.normalize(classifier_input, dim=-1)
            logits = self.head(classifier_input)

            mask = torch.zeros_like(logits)
            for i in range(batch_size):
                for j in range(num_tasks):
                    mask[i, j, self.class_mask[j]] = 1.

            logits = logits * mask
            logits = torch.sum(logits, dim=1)
            return logits, classifier_input, cls_feature, sim, tasks, classifier_input

        raise NotImplementedError(f'Unknown head_mode: {self.head_mode}')

    def forward(self, x, eval: bool = False):
        if self._is_ddp_head():
            logits, feats, frozen_feats, sim, tasks, classifier_input = self._forward_ddp_head(x, eval=eval)
            self._debug_print_shapes(
                input_image=x,
                patch_tokens=feats[:, 1:],
                cls_token=feats[:, 0:1],
                summarized_tokens=feats,
                task_tokens=feats,
                classifier_input=classifier_input,
                logits=logits,
            )
            return logits, feats, frozen_feats, sim, tasks

        token_features, task_feats, patch_tokens, cls_token = self.forward_features(x)
        logits, feats, frozen_feats, sim, tasks, classifier_input = self.forward_head(
            token_features, task_feats, eval=eval)
        self._debug_print_shapes(
            input_image=x,
            patch_tokens=patch_tokens,
            cls_token=cls_token,
            summarized_tokens=feats,
            task_tokens=task_feats,
            classifier_input=classifier_input,
            logits=logits,
        )
        return logits, feats, frozen_feats, sim, tasks

    def init(self, args):
        self.method = args.method
        self.t = 0
        self.num_tasks = args.num_tasks
        self.num_selectors = args.num_selectors
        self.normalize = args.normalize
        self.debug_shapes = bool(getattr(args, 'clip_debug_shapes', False))
        self.head_mode = args.head_mode
        if self._is_ddp_head():
            self._refresh_ddp_trainable_mask()
            return

        tokens = torch.randn((args.num_tasks, self.num_selectors, self.embed_dim))
        if args.prompt_init == 'orthogonal':
            tokens = nn.init.orthogonal_(tokens)
        elif args.prompt_init == 'uniform':
            tokens = nn.init.uniform_(tokens, -1, 1)
        self.selectors = nn.Parameter(tokens, requires_grad=True)

        if self.head_mode == 'learned_classifier':
            self.head_mode = 'concat'
        if self.head_mode == 'task':
            self.task_identifier = TaskIdentifier(args.num_tasks, self.embed_dim)

        for block in self.blocks:
            block.init(args)

    def next_task(self):
        self.t += 1
        if self._is_ddp_head():
            self._refresh_ddp_trainable_mask()
            if self.ddp_text_prompts.grad is not None:
                self.ddp_text_prompts.grad.zero_()
            if self.ddp_visual_prompts.grad is not None:
                self.ddp_visual_prompts.grad.zero_()
            return

        for block in self.blocks:
            block.next_task()

        with torch.no_grad():
            if hasattr(self, 'selectors'):
                if self.selectors.grad is not None:
                    self.selectors.grad.zero_()
                self.selectors[self.t] = self.selectors[self.t - 1]

        if hasattr(self, 'task_identifier'):
            self.task_identifier.next_task()
