
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
#from MedMamba.MedMamba import VSSM
import os 
#from MedViTV2.MedViT import MedViT_base
#from MedViTV2.MedViT import MedViT_large
#import open_clip
import json
from PIL import Image
from transformers import AutoProcessor, AutoModel



# ----------------- Tabular Component ----------------- #
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForwardGEGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.act = GEGLU()
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        x = self.out(x)
        return self.dropout(x)

class FTTransformerBlock(nn.Module):
    def __init__(self, dim, heads, attn_dropout, ffn_hidden, ffn_dropout, residual_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.residual_dropout1 = nn.Dropout(residual_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForwardGEGLU(dim, ffn_hidden, ffn_dropout)
        self.residual_dropout2 = nn.Dropout(residual_dropout)
    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.residual_dropout1(attn_out)
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + self.residual_dropout2(ffn_out)
        return x

class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, num_categories, token_dim, bias=True):
        super().__init__()
        self.embeddings = nn.Embedding(sum(num_categories), token_dim)
        self.bias = nn.Parameter(torch.zeros(len(num_categories), token_dim)) if bias else None
        category_offsets = torch.tensor([0] + num_categories[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets, persistent=False)

    def forward(self, x):
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, in_features, token_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, token_dim))
        self.bias = nn.Parameter(torch.zeros(in_features, token_dim)) if bias else None

    def forward(self, x):
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class FTTransformer(nn.Module):
    def __init__(
        self,
        num_numerical,
        num_categories,
        token_dim=192,
        hidden_size=192,
        num_blocks=3,
        attention_n_heads=8,
        attention_dropout=0.2,
        residual_dropout=0.0,
        ffn_dropout=0.1,
        ffn_hidden_size=192,
        pooling_mode="cls",
        num_classes=2, 
    ):
        super().__init__()
        # Tokenizers
        self.categorical_feature_tokenizer = CategoricalFeatureTokenizer(num_categories, token_dim) if num_categories else None
        self.numerical_feature_tokenizer = NumericalFeatureTokenizer(num_numerical, token_dim) if num_numerical else None

        # Adapters
        self.categorical_adapter = nn.Linear(token_dim, hidden_size) if num_categories else None
        self.numerical_adapter = nn.Linear(token_dim, hidden_size) if num_numerical else None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Transformer blocks
        self.transformer = nn.ModuleList([
            FTTransformerBlock(
                hidden_size,
                attention_n_heads,
                attention_dropout,
                ffn_hidden_size,
                ffn_dropout,
                residual_dropout
            )
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self.pooling_mode = pooling_mode

    def forward(self, batch, return_feature=False):
        # batch: expects dict with keys "categorical" and/or "numerical"
        B = batch['numerical'].shape[0] if 'numerical' in batch else batch['categorical'].shape[0]
        
        '''
        # to print the tabular input sizes
        if 'numerical' in batch:
            print(f"[FTTransformer] Numerical input shape: {batch['numerical'].shape}")
        if 'categorical' in batch:
            print(f"[FTTransformer] Categorical input shape: {batch['categorical'].shape}")
        '''

        multimodal_tokens = []

        if self.categorical_feature_tokenizer:
            x_cat = self.categorical_feature_tokenizer(batch['categorical'])  # (B, num_cat, token_dim)
            x_cat = self.categorical_adapter(x_cat)
            multimodal_tokens.append(x_cat)

        if self.numerical_feature_tokenizer:
            x_num = self.numerical_feature_tokenizer(batch['numerical'])  # (B, num_num, token_dim)
            x_num = self.numerical_adapter(x_num)
            multimodal_tokens.append(x_num)

        tokens = torch.cat(multimodal_tokens, dim=1)  # (B, total_num_tokens, hidden_size)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
        tokens = torch.cat([tokens, cls_token], dim=1)  # (B, total_num_tokens+1, hidden_size)

        for block in self.transformer:
            tokens = block(tokens)

        features = tokens[:, -1, :]
        logits = self.head(features)

        if return_feature:
            return features, logits
        return logits
    
# ----------------- Image Component (Swin Transformer) ----------------- #
class TimmModel(nn.Module):
    def __init__(self, num_classes=2, max_img_num_per_col=3, use_learnable_image=True):
        super().__init__()
        self.max_img_num_per_col = max_img_num_per_col
        self.use_learnable_image = use_learnable_image

        self.base_model = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=True,
            num_classes=0
        )
        self.feature_dim = self.base_model.num_features  # 1024 for swin_base_patch4_window7_224
        self.head = nn.Linear(self.feature_dim, num_classes)

        if use_learnable_image:
            self.learnable_image = nn.Parameter(torch.zeros(3, 224, 224))

    def forward(self, x, image_valid_num=None, return_feature=False):
        """
        x: [B, N, C, H, W]
        image_valid_num: optional, number of valid images per sample
        """
        B, N, C, H, W = x.shape
        device = x.device

        # Flatten batch and planes
        x_flat = x.view(B * N, C, H, W)

        # 🔑 Use forward_features() to get embeddings
        feats = self.base_model.forward_features(x_flat)  # shape [B*N, feature_dim]

        # Reshape to [B, N, feature_dim]
        feats = feats.view(B, N, -1)
        return logits

# ----------------- Image Component (MedMamba) ----------------- #
class MedMambaModel(nn.Module):
    def __init__(self, num_classes=2, max_img_num_per_col=3, use_learnable_image=True, 
                 model_variant='b', weights_path=None, device='cuda'):
        super().__init__()
        self.max_img_num_per_col = max_img_num_per_col
        self.use_learnable_image = use_learnable_image

        # Define MedMamba variant
        if model_variant == 'b':
            self.base_model = VSSM(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=0)
        elif model_variant == 's':
            self.base_model = VSSM(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=0)
        elif model_variant == 't':
            self.base_model = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=0)
        else:
            raise ValueError(f"Unknown model_variant: {model_variant}")

        # Load pretrained weights (optional)
        if weights_path is not None and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device)
            state_dict = checkpoint.get('model', checkpoint)
            self.base_model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded MedMamba weights from {weights_path}")

        self.feature_dim = self.base_model.num_features
        self.head = nn.Linear(self.feature_dim, num_classes)

        if use_learnable_image:
            self.learnable_image = nn.Parameter(torch.zeros(3, 224, 224))

    def forward(self, x, image_valid_num=None, return_feature=False):
        B, N, C, H, W = x.shape
        device = x.device

        # Fill missing images with learnable tensor
        if self.use_learnable_image and image_valid_num is not None:
            for b in range(B):
                n_valid = image_valid_num[b].item()
                if n_valid < N:
                    x[b, n_valid:] = self.learnable_image

        x_flat = x.reshape(B * N, C, H, W)
        feats = self.base_model.forward_backbone(x_flat)
        feats = feats.mean(dim=(1, 2))  # GAP over H, W
        feats = feats.view(B, N, -1)

        # Mask invalid images
        if image_valid_num is None:
            image_valid_num = torch.full((B,), N, dtype=torch.long, device=device)
        mask = torch.arange(N, device=device)[None, :] < image_valid_num[:, None]
        feats = feats * mask.unsqueeze(-1)
        avg_feats = feats.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        logits = self.head(avg_feats)
        return (avg_feats, logits) if return_feature else logits
    
# ----------------- Image Component (MedViTV2) ----------------- #
class MedViTModel(nn.Module):
    def __init__(self, variant='base', pretrained_path=None, freeze=False, device='cpu'):
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

                # Some checkpoints save as {'model': state_dict} or have prefixes
                if 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt

                # --- 🔧 Clean up key names ---
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

                # --- ✅ Load cleaned weights ---
                missing, unexpected = self.base_model.load_state_dict(new_state_dict, strict=False)
                total_params = sum(p.numel() for p in self.base_model.parameters())
                loaded_params = sum(v.numel() for k, v in new_state_dict.items() if k in self.base_model.state_dict())

                print(f"✅ Loaded MedViT-{variant} pretrained weights from {pretrained_path}")
                print(f"Loaded params: {loaded_params/1e6:.2f}M / {total_params/1e6:.2f}M "
                    f"({100 * loaded_params/total_params:.2f}% of model)")
                print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                print("Example missing keys:", missing[:20])

            else:
                print(f"⚠️ WARNING: Pretrained path not found: {pretrained_path}. Using random init.")
        else:
            print("⚠️ No pretrained weights used (pretrained_path=None).")



        # Optionally freeze backbone
        if freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False

        self.output_dim = output_dim
        self.proj = nn.Linear(output_dim, output_dim)
        self.feature_dim = output_dim
        self.device = device
        self.to(device)

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

        return (feats, None) if return_feature else feats
        
# ----------------- Image Component (FetalCLIP) ----------------- #
class FetalCLIPModel(nn.Module):
    def __init__(self, config_path="FetalCLIP/FetalCLIP_config.json", weight_path="FetalCLIP_weights.pt", freeze=True, device='cuda'):
        super().__init__()

        # Load model config and register it with open_clip
        with open(config_path, "r") as f:
            config = json.load(f)
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config

        # Create the model and its transforms
        self.model, self.preprocess_train, self.preprocess_test = open_clip.create_model_and_transforms(
            "FetalCLIP", pretrained=weight_path
        )

        self.device = device
        self.model.to(device)
        self.model.eval()

        # Optional projection layer to match fusion dims
        self.feature_dim = self.model.visual.output_dim
        self.proj = nn.Linear(self.feature_dim, self.feature_dim).to(device)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x, image_valid_num=None, return_feature=False):
        """
        x: [B, N, C, H, W] — preprocessed tensor images (use preprocess_test)
        image_valid_num: optional number of valid images per sample
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = self.proj(feats)
            feats = feats.view(B, N, -1)

        # Average across valid images
        if image_valid_num is not None:
            mask = torch.arange(N, device=self.device).unsqueeze(0) < image_valid_num.unsqueeze(1)
            feats = (feats * mask.unsqueeze(-1)).sum(dim=1) / image_valid_num.unsqueeze(1).clamp(min=1)
        else:
            feats = feats.mean(dim=1)
        
        feats = feats.float()

        if return_feature:
            return feats, None
        return feats
    
# ----------------- Image Component (MedSigLIP) ----------------- #
class MedSigLIPModel(nn.Module):
    """
    Fine-tuneable MedSigLIP vision encoder for multimodal training.
    """

    def __init__(self, model_name="google/medsiglip-448", device=None, freeze=False, unfreeze_last_n_blocks=2):
        super().__init__()
        from transformers import AutoProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model + processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

        self.feature_dim = self.model.vision_model.config.hidden_size

        self.model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled for MedSigLIP")


        # Optional freezing
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            # Optionally freeze most layers except last N blocks
            total_blocks = len(self.model.vision_model.encoder.layers)
            for i, block in enumerate(self.model.vision_model.encoder.layers):
                if i < total_blocks - unfreeze_last_n_blocks:
                    for p in block.parameters():
                        p.requires_grad = False

        print(f"✅ MedSigLIP loaded. Fine-tuning last {unfreeze_last_n_blocks} blocks.")
        
    def forward(self, x, image_valid_num=None, return_feature=False):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        # Normalize as expected by MedSigLIP
        outputs = self.model.vision_model(pixel_values=x)
        feats = outputs.pooler_output  # [B*N, feature_dim]
        feats = feats.view(B, N, -1)

        # Average across planes
        if image_valid_num is not None:
            mask = torch.arange(N, device=self.device)[None, :] < image_valid_num[:, None]
            feats = (feats * mask.unsqueeze(-1)).sum(dim=1) / image_valid_num.unsqueeze(1).clamp(min=1)
        else:
            feats = feats.mean(dim=1)

        if return_feature:
            return feats, None
        return feats


# ----------------- Fusion Component ----------------- #
class FusionMLPBlock(nn.Module):
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

class FusionMLP(nn.Module):
    def __init__(
        self,
        tabular_dim,      # output dim of FTTransformer before its head (not logits!)
        image_dim,        # output dim of TimmModel before its head (not logits!)
        hidden_sizes=[128],
        num_classes=2,
        adapt_in_features='max',
        activation='leaky_relu',
        dropout=0.1,
        normalization='layer_norm',
    ):
        super().__init__()
        # Feature adaptation
        if adapt_in_features == 'max':
            base_dim = max(tabular_dim, image_dim)
        elif adapt_in_features == 'min':
            base_dim = min(tabular_dim, image_dim)
        else:
            raise ValueError(f"Unknown adapt_in_features: {adapt_in_features}")
        self.tabular_adapter = nn.Linear(tabular_dim, base_dim)
        self.image_adapter = nn.Linear(image_dim, base_dim)
        fusion_in_dim = 2 * base_dim

        # MLP layers
        layers = []
        for h in hidden_sizes:
            layers.append(FusionMLPBlock(
                in_features=fusion_in_dim,
                out_features=h,
                activation=activation,
                dropout=dropout,
                normalization=normalization
            ))
            fusion_in_dim = h
        self.fusion_mlp = nn.Sequential(*layers)
        self.head = nn.Linear(fusion_in_dim, num_classes)
    
    def forward(self, tabular_features, image_features):
        # tabular_features: [B, tabular_dim], image_features: [B, image_dim]
        #print(f"Tabular embedding shape: {tabular_features.shape}")
        #print(f"Image embedding shape: {image_features.shape}")

        tabular_proj = self.tabular_adapter(tabular_features)
        image_proj = self.image_adapter(image_features)

        #print(f"Tabular projected shape: {tabular_proj.shape}")
        #print(f"Image projected shape: {image_proj.shape}")

        fused = torch.cat([tabular_proj, image_proj], dim=1)
        #print(f"Fused shape: {fused.shape}")

        fused = self.fusion_mlp(fused)
        logits = self.head(fused)
        return logits
    '''
    def forward(self, inputs):
        # tabular_features: [B, tabular_dim], image_features: [B, image_dim]
        tabular_features = inputs["tabular"]
        image_features = inputs["image"]

        tabular_proj = self.tabular_adapter(tabular_features)
        image_proj = self.image_adapter(image_features)

        #print(f"Tabular projected shape: {tabular_proj.shape}")
        #print(f"Image projected shape: {image_proj.shape}")

        fused = torch.cat([tabular_proj, image_proj], dim=1)
        #print(f"Fused shape: {fused.shape}")

        fused = self.fusion_mlp(fused)
        logits = self.head(fused)
        return logits
    '''

# -------------- Final Model with all Components -------------- #
class MultimodalModel(nn.Module):
    def __init__(self, fttransformer, image_encoder, fusion_mlp):
        super().__init__()
        self.fttransformer = fttransformer
        self.image_encoder = image_encoder
        self.fusion_mlp = fusion_mlp
    
    def forward(self, batch):
        # batch: expects dict with keys "numerical", "categorical", "images", and optionally "image_valid_num"
        tabular_feat, _ = self.fttransformer(batch, return_feature=True)
        image_feat, _ = self.image_encoder(
            batch["images"],
            image_valid_num=batch.get("image_valid_num"),
            return_feature=True
        )
      
        tabular_feat = F.layer_norm(tabular_feat, tabular_feat.shape[1:])
        image_feat = F.layer_norm(image_feat, image_feat.shape[1:])
        
        logits = self.fusion_mlp(tabular_feat, image_feat)
        return logits
    

    
def build_multimodal_model(num_numerical, num_categories, tabular_token_dim, tabular_hidden_dim):
    fttransformer = FTTransformer(
        num_numerical=num_numerical,
        num_categories=num_categories,
        token_dim=tabular_token_dim,
        hidden_size=tabular_hidden_dim,
        num_blocks=3,
        attention_n_heads=8,
        num_classes=2 
    )

    image_model = TimmModel(num_classes=2) 

    '''
    image_model = MedMambaModel(
    num_classes=2,
    model_variant='s',
    #weights_path=None, 
    weights_path='MedMamba.pth',  
    device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    image_model = MedViTModel(
        variant = 'base',
        pretrained_path="MedViT_base_Fetal.pth",  # path to your weights
        #pretrained_path= None,
        #output_dim=768,
        freeze=False, 
        device='cuda' if torch.cuda.is_available() else 'cpu')
    
    
    image_model = FetalCLIPModel(
        config_path="FetalCLIP/FetalCLIP_config.json",
        weight_path="FetalCLIP_weights.pt",
        freeze=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    image_model = MedSigLIPModel(
    model_name="google/medsiglip-448",
    device='cuda' if torch.cuda.is_available() else 'cpu',
    freeze=False,                 #  False for trainable and True for non trainable
    unfreeze_last_n_blocks=2      # only fine-tune last 2 transformer blocks
    )'''

    
    fusion_mlp = FusionMLP(
        tabular_dim=tabular_hidden_dim,
        image_dim=image_model.feature_dim, #Change here if needed
        hidden_sizes=[128],
        num_classes=2
    )
    multimodal_model = MultimodalModel(fttransformer, image_model, fusion_mlp)
    return multimodal_model


