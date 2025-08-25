import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
import loralib as lora
import inspect


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, lora_rank=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if lora_rank > 0:
            self.fc1 = lora.Linear(in_features, hidden_features, r=lora_rank)
            self.fc2 = lora.Linear(hidden_features, out_features, r=lora_rank)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 lora_rank=0, lora_pos="FFN", block_index=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            lora_rank=lora_rank if lora_pos == "FFN" else 0
        )
        self.group_name = f"group_{block_index}" if block_index is not None else None

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# class VisionTransformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
#                  embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, qkv_bias=True,
#                  drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
#                  use_lora=False, lora_rank=4):

#         super().__init__()

#         # 🔍 Step 1: Get the default args from the function signature
#         frame = inspect.currentframe()
#         args, _, _, values = inspect.getargvalues(frame)
#         signature = inspect.signature(self.__init__)
#         default_args = {k: v.default for k, v in signature.parameters.items()}

#         # 🔍 Step 2: Identify non-default arguments passed
#         explicitly_passed = {
#             arg: values[arg] for arg in args
#             if arg != 'self' and values[arg] != default_args.get(arg)
#         }

#         if explicitly_passed:
#             print(f"🛠️ Explicitly received params: {explicitly_passed}")
#         else:
#             print("✅ No non-default parameters were passed. Using all defaults.")

#         # 🌱 Usual model init below
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim

#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.blocks = nn.Sequential(*[
#             Block(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[i],
#                 lora_rank=lora_rank if use_lora else 0,
#                 lora_pos="FFN",
#                 block_index=i
#             )
#             for i in range(depth)
#         ])

#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         self._init_weights()

#         if use_lora and lora_rank > 0:
#             self.freeze_all_but_lora()


#     def _init_weights(self):
#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_module)

#     def _init_module(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)

#     def freeze_all_but_lora(self):
#         print("🔒 Freezing all model parameters except LoRA and classifier head...")
#         for name, param in self.named_parameters():
#             if 'lora_' in name or 'head' in name:
#                 param.requires_grad = True
#                 # print(f"✅ Trainable: {name}")
#             else:
#                 param.requires_grad = False

#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         x = self.blocks(x)
#         x = self.norm(x)

#         cls_output = x[:, 0]
#         return self.head(cls_output)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, qkv_bias=True,
                 drop_rate=0, attn_drop_rate=0, drop_path_rate=0, lora_rank=0):

        super().__init__()

        # 🔍 Step 1: Get the default args from the function signature
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        signature = inspect.signature(self.__init__)
        default_args = {k: v.default for k, v in signature.parameters.items()}

        # 🔍 Step 2: Identify non-default arguments passed
        explicitly_passed = {
            arg: values[arg] for arg in args
            if arg != 'self' and values[arg] != default_args.get(arg)
        }

        if explicitly_passed:
            print(f"🛠️ Explicitly received params: {explicitly_passed}")
        else:
            print("✅ No non-default parameters were passed. Using all defaults.")

        # 🌱 Usual model init below
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.lora_rank = lora_rank  # Store lora_rank for _configure_trainable_params

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                lora_rank=lora_rank,  # Pass lora_rank directly
                lora_pos="FFN",
                block_index=i
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()
        self._configure_trainable_params()  # Configure trainable params based on lora_rank

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _configure_trainable_params(self):
        """
        Freeze/unfreeze based on LoRA rank.
        - rank=0  → full fine-tuning (all params trainable).
        - rank>0  → freeze backbone, unfreeze LoRA + classifier.
        """
        if self.lora_rank == 0:
            # full fine-tuning
            print("🔓 Full fine-tuning: All parameters trainable")
            for _, p in self.named_parameters():
                p.requires_grad = True
        else:
            print(f"🔒 LoRA mode (rank={self.lora_rank}): Freezing backbone, unfreezing LoRA + classifier")
            # freeze all
            for _, p in self.named_parameters():
                p.requires_grad = False
            # unfreeze LoRA params
            for name, p in self.named_parameters():
                if "lora_" in name:  # LoRA layers
                    p.requires_grad = True
            # unfreeze classifier head
            for name, p in self.named_parameters():
                if "head" in name:
                    p.requires_grad = True

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        cls_output = x[:, 0]
        return self.head(cls_output)