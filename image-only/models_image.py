import torch
import torch.nn as nn
import timm
import os
import json
from transformers import AutoProcessor, AutoModel
import sys 
sys.path.append(os.path.abspath("../multimodal"))
import MedViTV2


#  Shared MLP Head (same logic as FusionMLPBlock)

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='leaky_relu', dropout=0.1, normalization='layer_norm'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features) if normalization == 'layer_norm' else nn.Identity()
        self.act = nn.LeakyReLU() if activation == 'leaky_relu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ImageMLPHead(nn.Module):
    def __init__(self, in_dim, hidden_sizes=[128], num_classes=2, dropout=0.2):
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden_sizes:
            layers.append(MLPBlock(dim, h, dropout=dropout))
            dim = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.mlp(x)
        logits = self.head(x)
        return logits



#  Swin Transformer

class SwinModel(nn.Module):
    def __init__(self, num_classes=2, max_img_num=3, freeze=False, use_learnable_image=False):
        super().__init__()
        self.max_img_num = max_img_num
        self.use_learnable_image = use_learnable_image

        self.base_model = timm.create_model(
            "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            pretrained=True,
            num_classes=0
        )
        self.feature_dim = self.base_model.num_features

   
        if freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False
            print("Swin backbone frozen.")

        if use_learnable_image:
            self.learnable_image = nn.Parameter(torch.zeros(3, 224, 224))

        self.head = ImageMLPHead(self.feature_dim, [128], num_classes)

    def forward(self, x, image_valid_num=None):
        B, N, C, H, W = x.shape
        device = x.device

        if self.use_learnable_image and image_valid_num is not None:
            for b in range(B):
                n_valid = image_valid_num[b].item()
                if n_valid < N:
                    x[b, n_valid:] = self.learnable_image

        x = x.reshape(B * N, C, H, W)
        feats = self.base_model(x)                      # [B*N, feature_dim]
        feats = feats.view(B, N, -1)

        if image_valid_num is None:
            image_valid_num = torch.full((B,), N, dtype=torch.long, device=device)

        mask = torch.arange(N, device=device)[None, :] < image_valid_num[:, None]
        feats = feats * mask.unsqueeze(-1)
        avg_feats = feats.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        logits = self.head(avg_feats)
        return logits



#  MedViT

class MedViTModel(nn.Module):
    def __init__(self, num_classes=2, variant='base', pretrained_path=None, freeze=False, device='cpu'):
        super().__init__()

        # Dynamically select model variant
        if variant == 'base':
            from MedViTV2.MedViT import MedViT_base
            self.base_model = MedViT_base(pretrained=False)
            output_dim = 768
        elif variant == 'large':
            from MedViTV2.MedViT import MedViT_large
            self.base_model = MedViT_large(pretrained=False)
            output_dim = 1024
        else:
            raise ValueError(f"Unknown MedViT variant: {variant}. Choose 'base' or 'large'.")

        # Core feature extractor (remove classifier)
        self.feature_extractor = nn.Sequential(
            self.base_model.stem,
            self.base_model.features,
            self.base_model.norm,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.base_model.proj_head = nn.Identity()

        # Optionally load pretrained weights
        if pretrained_path is not None:
            if os.path.exists(pretrained_path):
                ckpt = torch.load(pretrained_path, map_location='cpu', weights_only = True)

              
                if 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt

               
                new_state_dict = {}
                for k, v in state_dict.items():
                    k_new = k
                    # Strip common prefixes
                    if k.startswith("model."):
                        k_new = k[len("model."):]
                    if k.startswith("module."):
                        k_new = k[len("module."):]
                    if k.startswith("encoder."):
                        k_new = k[len("encoder."):]
                    # Drop classifier / projection heads
                    if "head" in k_new or "proj_head" in k_new:
                        continue
                    new_state_dict[k_new] = v

                # Load weights 
                missing, unexpected = self.base_model.load_state_dict(new_state_dict, strict=False)
                total_params = sum(p.numel() for p in self.base_model.parameters())
                loaded_params = sum(v.numel() for k, v in new_state_dict.items() if k in self.base_model.state_dict())

                print(f"Loaded MedViT-{variant} pretrained weights from {pretrained_path}")
                print(f"Loaded params: {loaded_params/1e6:.2f}M / {total_params/1e6:.2f}M "
                    f"({100 * loaded_params/total_params:.2f}% of model)")
                print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                print("Example missing keys:", missing[:20])

            else:
                print(f"WARNING: Pretrained path not found: {pretrained_path}. Using random init.")
        else:
            print("No pretrained weights used (pretrained_path=None).")



        # Optionally freeze backbone
        if freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False

        self.output_dim = output_dim
        self.proj = nn.Linear(output_dim, output_dim)
        self.feature_dim = output_dim
        self.device = device
        self.to(device)

        self.head = ImageMLPHead(self.feature_dim, [128], num_classes=num_classes)

    def forward(self, x, image_valid_num=None, return_feature=False):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        feats = self.proj(feats)
        feats = feats.view(B, N, -1)

        if image_valid_num is not None:
            mask = torch.arange(N, device=self.device).unsqueeze(0) < image_valid_num.unsqueeze(1)
            feats = (feats * mask.unsqueeze(-1)).sum(dim=1) / image_valid_num.unsqueeze(1).clamp(min=1)
        else:
            feats = feats.mean(dim=1)

        if return_feature:
            return feats, None

        
        logits = self.head(feats)
        return logits



#  MedMamba

class MedMambaModel(nn.Module):
    def __init__(self, num_classes=2, model_variant='s', weights_path=None, freeze=False, device='cuda'):
        super().__init__()
        from MedMamba.MedMamba import VSSM

        # Choose variant
        if model_variant == 'b':
            self.base_model = VSSM(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=0)
        elif model_variant == 's':
            self.base_model = VSSM(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=0)
        elif model_variant == 't':
            self.base_model = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=0)
        else:
            raise ValueError(f"Unknown MedMamba variant: {model_variant}")

        # Load pretrained weights
        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = ckpt.get('model', ckpt)
            self.base_model.load_state_dict(state_dict, strict=False)
            print(f"Loaded MedMamba-{model_variant} weights from {weights_path}")
        else:
            print("No MedMamba weights loaded (using random init).")

       
        self.base_model.to(device)

        if freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False
            print("MedMamba backbone frozen.")

        # Detect actual feature dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            dummy_out = self.base_model.forward_backbone(dummy)
            print("Detected MedMamba backbone output shape:", dummy_out.shape)
            if dummy_out.ndim == 4:
                feature_dim = dummy_out.shape[1]
            else:
                feature_dim = dummy_out.shape[-1]

        self.feature_dim = feature_dim
        print(f"Using feature_dim={feature_dim} for MLP head")

        self.head = ImageMLPHead(self.feature_dim, [128], num_classes)
        self.device = device


    def forward(self, x, image_valid_num=None):
        B, N, C, H, W = x.shape
        x_flat = x.reshape(B * N, C, H, W)

        feats = self.base_model.forward_backbone(x_flat)
        
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))

        feats = feats.view(B, N, -1)
        mask = torch.arange(N, device=self.device)[None, :] < image_valid_num[:, None]
        feats = feats * mask.unsqueeze(-1)
        avg_feats = feats.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        logits = self.head(avg_feats)
        return logits




#  FetalCLIP

class FetalCLIPModel(nn.Module):
    def __init__(self, config_path, weight_path, num_classes=2, device='cuda', freeze=True):
        super().__init__()
        import open_clip
        with open(config_path, "r") as f:
            config = json.load(f)
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config

        self.model, _, _ = open_clip.create_model_and_transforms("FetalCLIP", pretrained=weight_path)
        self.model.to(device)
        '''
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        '''
        if freeze:
            for name, p in self.model.named_parameters():
            # Unfreeze last 4 blocks (20,21,22,23)
                if not any(k in name for k in [
                    "visual.transformer.resblocks.23", 
                    "visual.transformer.resblocks.22",
                    "visual.transformer.resblocks.21",
                    "visual.transformer.resblocks.20"
                ]):
                    p.requires_grad = False
            print("Only last 4 layers of FetalCLIP unfrozen.")

        self.model.eval()

        self.feature_dim = self.model.visual.output_dim
        self.proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.head = ImageMLPHead(self.feature_dim, [128], num_classes)
        self.device = device

    def forward(self, x, image_valid_num=None):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        with torch.no_grad():
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        feats = self.proj(feats)
        feats = feats.view(B, N, -1)

        mask = torch.arange(N, device=self.device)[None, :] < image_valid_num[:, None]
        feats = feats * mask.unsqueeze(-1)
        avg_feats = feats.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        logits = self.head(avg_feats)
        return logits



#  MedSigLIP

class MedSigLIPModel(nn.Module):
    def __init__(self, model_name="google/medsiglip-448", num_classes=2, device='cuda',
                 freeze=False, unfreeze_last_n_blocks=0):
        super().__init__()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.gradient_checkpointing_enable()

        total_blocks = len(self.model.vision_model.encoder.layers)
        print(f"MedSigLIP has {total_blocks} transformer blocks.")


        if freeze and unfreeze_last_n_blocks == 0:
            # Fully frozen
            for p in self.model.parameters():
                p.requires_grad = False
            print("MedSigLIP backbone fully frozen.")
        elif freeze and unfreeze_last_n_blocks > 0:
            # Freeze all except last N
            for i, block in enumerate(self.model.vision_model.encoder.layers):
                if i < total_blocks - unfreeze_last_n_blocks:
                    for p in block.parameters():
                        p.requires_grad = False
            print(f"Fine-tuning last {unfreeze_last_n_blocks} blocks of MedSigLIP.")
        else:
            print("MedSigLIP backbone fully trainable.")

        # Feature and classification head 
        self.feature_dim = self.model.vision_model.config.hidden_size
        self.head = ImageMLPHead(self.feature_dim, [128], num_classes)
        self.device = device

    def forward(self, x, image_valid_num=None):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        outputs = self.model.vision_model(pixel_values=x)
        feats = outputs.pooler_output
        feats = feats.view(B, N, -1)

        mask = torch.arange(N, device=self.device)[None, :] < image_valid_num[:, None]
        feats = feats * mask.unsqueeze(-1)
        avg_feats = feats.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        logits = self.head(avg_feats)
        return logits




#  Build Model

def build_image_model(
    encoder_name,
    num_classes=2,
    device='cuda',
    pretrained=True,
    pretrained_path=None,
    freeze=False,
    model_variant='base'
):
    
    encoder_name = encoder_name.lower()

    if encoder_name == "swin":
        return SwinModel(
            num_classes=num_classes,
            freeze=freeze,
            use_learnable_image=False
        )

    elif encoder_name == "medvit_pt":
        return MedViTModel(
            num_classes= num_classes,
            variant=model_variant,
            pretrained_path=pretrained_path,
            freeze=freeze,
            device=device
        )
    
    elif encoder_name == "medvit_nopt":
        return MedViTModel(
            num_classes= num_classes,
            variant=model_variant,
            pretrained_path=None,
            freeze=freeze,
            device=device
        )

    elif encoder_name == "medmamba":
        return MedMambaModel(
            num_classes=num_classes,
            model_variant=model_variant,
            weights_path=pretrained_path,
            freeze=freeze,
            device=device
        )

    elif encoder_name == "fetalclip_2":
        return FetalCLIPModel(
            config_path="../multimodal/FetalCLIP/FetalCLIP_config.json",
            weight_path=pretrained_path if pretrained_path else "../multimodal/FetalCLIP_weights.pt",
            num_classes=num_classes,
            device=device,
            freeze=freeze
        )

    elif encoder_name == "medsiglip_frozen":
        return MedSigLIPModel(
            model_name="google/medsiglip-448",
            num_classes=num_classes,
            device=device,
            freeze=True,
            unfreeze_last_n_blocks=0  # fully frozen
        )

    elif encoder_name == "medsiglip_finetune":
        return MedSigLIPModel(
            model_name="google/medsiglip-448",
            num_classes=num_classes,
            device=device,
            freeze=True,                   # freeze most layers
            unfreeze_last_n_blocks=2        # fine-tune last 2 blocks
        )

    else:
        raise ValueError(f" Unknown encoder: {encoder_name}")

