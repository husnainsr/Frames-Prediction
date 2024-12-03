import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, seq_length):
        super(TransformerModel, self).__init__()
        self.name = "Transformer"
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # CNN layers for spatial feature extraction
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * 64 * 64, hidden_dim)
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 64 * 64),
            nn.Sigmoid()
        )

    def forward(self, src):
        # src shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, channels, height, width = src.size()
        
        # Process each frame through CNN
        # Reshape to process all frames at once
        src = src.view(-1, channels, height, width)
        features = self.cnn_encoder(src)
        features = features.view(batch_size, seq_length, self.hidden_dim)

        # Add positional encoding
        features = self.positional_encoding(features)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(features)

        # Decode the last 5 positions to generate predictions
        output_sequence = transformer_output[:, -5:, :]
        
        # Decode each timestep
        decoded_frames = []
        for t in range(output_sequence.size(1)):
            decoded = self.decoder(output_sequence[:, t])
            decoded = decoded.view(batch_size, 1, 1, height, width)
            decoded_frames.append(decoded)
        
        # Concatenate all frames
        output = torch.cat(decoded_frames, dim=1)
        return output