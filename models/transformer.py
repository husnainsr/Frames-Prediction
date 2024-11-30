import torch
import torch.nn as nn
import math

class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Conv2d(d_model, d_model, 1)
        self.key = nn.Conv2d(d_model, d_model, 1)
        self.value = nn.Conv2d(d_model, d_model, 1)
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Simple scaled dot-product attention
        q_flat = q.flatten(2)
        k_flat = k.flatten(2)
        v_flat = v.flatten(2)
        
        attn = torch.matmul(q_flat, k_flat.transpose(-2, -1))
        attn = attn / math.sqrt(self.d_model)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v_flat)
        out = out.view_as(x)
        return out

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = SimpleAttention(d_model)
        self.norm1 = nn.LayerNorm([d_model])
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, 1),
            nn.ReLU(),
            nn.Conv2d(d_model * 2, d_model, 1)
        )
        self.norm2 = nn.LayerNorm([d_model])
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Attention
        attended = self.attention(x)
        x = x + attended
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, input_channels=1, d_model=32, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Conv2d(input_channels, d_model, 3, padding=1)
        
        self.encoder_layers = nn.ModuleList([
            SimpleTransformerBlock(d_model)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            SimpleTransformerBlock(d_model)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(d_model, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, src, tgt, training=True):
        # Print input shapes
        # print(f"Source shape: {src.shape}")
        # print(f"Target shape: {tgt.shape}")
        
        batch_size = src.shape[0]
        src_time_steps = src.shape[1]
        tgt_time_steps = tgt.shape[1]
        
        # Make tensors contiguous and reshape
        src_flat = src.contiguous().reshape(-1, *src.shape[2:])  # [B*T, C, H, W]
        src_emb = self.embedding(src_flat)  # [B*T, d_model, H, W]
        
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)
        
        # Process target sequence
        tgt_flat = tgt.contiguous().reshape(-1, *tgt.shape[2:])  # [B*T, C, H, W]
        tgt_emb = self.embedding(tgt_flat)  # [B*T, d_model, H, W]
        
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb)
        
        # Final prediction
        out = self.final_layer(tgt_emb)  # [B*T, C, H, W]
        out = out.reshape(batch_size, tgt_time_steps, *out.shape[1:])  # [B, T, C, H, W]
        
        return out, None
    
    @property
    def name(self):
        return "SimpleTransformer"