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
        self.debug_shapes = bool(getattr(args, 'clip_debug_shapes', False))
        self._debug_shapes_printed = False
        self.train_logit_scale = bool(getattr(args, 'train_logit_scale', False))
        self.text_templates = self._parse_text_templates(args)
        self.class_names = None
        self._text_feature_cache_key = None
        self._text_feature_class_ids = tuple()
        self._last_clip_text_debug_shapes = None
        self._debug_text_cache_keys_printed = set()
        self.register_buffer('text_features', torch.empty(0), persistent=False)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        clip_blocks = self.visual.transformer.resblocks
        act_layer = act_layer or type(clip_blocks[0].mlp.gelu)
        depth = len(clip_blocks)
        num_heads = clip_blocks[0].attn.num_heads
        mlp_ratio = 4.
        qkv_bias = True
        pret_attention = True if args is None else args.method in ['prompts']
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

        tokens = torch.randn((args.num_tasks, self.num_selectors, self.embed_dim))
        if args.prompt_init == 'orthogonal':
            tokens = nn.init.orthogonal_(tokens)
        elif args.prompt_init == 'uniform':
            tokens = nn.init.uniform_(tokens, -1, 1)
        self.selectors = nn.Parameter(tokens, requires_grad=True)

        self.head_mode = args.head_mode
        if self.head_mode == 'learned_classifier':
            self.head_mode = 'concat'
        if self.head_mode == 'task':
            self.task_identifier = TaskIdentifier(args.num_tasks, self.embed_dim)

        for block in self.blocks:
            block.init(args)

    def next_task(self):
        self.t += 1
        for block in self.blocks:
            block.next_task()

        with torch.no_grad():
            if hasattr(self, 'selectors'):
                if self.selectors.grad is not None:
                    self.selectors.grad.zero_()
                self.selectors[self.t] = self.selectors[self.t - 1]

        if hasattr(self, 'task_identifier'):
            self.task_identifier.next_task()
