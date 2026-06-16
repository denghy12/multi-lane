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
        self.visual = self.clip_model.visual
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        self.num_features = self.embed_dim = self.visual.conv1.weight.shape[0]
        self.num_prefix_tokens = 1
        self.grad_checkpointing = False
        self.debug_shapes = bool(getattr(args, 'clip_debug_shapes', False))
        self._debug_shapes_printed = False

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

        self._load_clip_visual_transformer_weights()
        self.head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        self._debug_shapes_printed = True

    def forward_head(self, token_features, task_tokens, eval: bool = False):
        cls_feature = token_features[:, 0]
        device = cls_feature.device
        batch_size = cls_feature.size(0)
        sim = torch.ones((), device=device, dtype=cls_feature.dtype)
        tasks = torch.zeros(batch_size, dtype=torch.long, device=device)

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

        if self.head_mode == 'concat':
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
