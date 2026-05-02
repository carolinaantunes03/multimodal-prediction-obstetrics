
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
    
