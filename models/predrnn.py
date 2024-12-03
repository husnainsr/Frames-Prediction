import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=5):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.input_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Gates
        total_channels = in_channels + hidden_channels
        self.gates = nn.Sequential(
            nn.Conv2d(total_channels, 4 * hidden_channels, kernel_size, padding=padding),
            nn.GroupNorm(4, 4 * hidden_channels)
        )

        # Memory state update
        self.memory_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.GroupNorm(4, hidden_channels)
        )

        # Fix: Update residual connection to match input channels
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GroupNorm(4, hidden_channels)
        )

    def forward(self, x, h, c, m):
        if h is None:
            batch_size, _, height, width = x.size()
            device = x.device
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            m = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)

        # Apply residual connection
        res = self.residual(x)

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Calculate gates
        gates = self.gates(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # Update memory state
        m_transformed = self.memory_conv(m)
        c_next = f * c + i * g + torch.tanh(m_transformed)
        h_next = o * torch.tanh(c_next) + res

        # Update memory gate
        m_next = torch.tanh(self.memory_conv(h_next))

        return h_next, c_next, m_next

class PredRNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, num_frames_output, name="PredRNN"):
        super(PredRNN, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.num_frames_output = num_frames_output

        # Input projection
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.LeakyReLU(0.2)
        )

        # LSTM cells
        self.cell_list = nn.ModuleList([
            SpatioTemporalLSTMCell(
                in_channels=hidden_channels * 2 if i == 0 else hidden_channels,
                hidden_channels=hidden_channels
            ) for i in range(num_layers)
        ])

        # Detail preservation
        self.detail_preserve = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(4, hidden_channels),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_layers)
        ])

        # Modified output projection with brightness preservation
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_channels + hidden_channels * 2, hidden_channels * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels * 2, input_channels, 3, padding=1),
            nn.Sigmoid(),  # Keep sigmoid but add brightness adjustment
        )

    def forward(self, frames, num_predictions=None):
        if num_predictions is None:
            num_predictions = self.num_frames_output

        batch_size, seq_length, _, height, width = frames.size()

        # Initialize states
        h_t = [None] * self.num_layers
        c_t = [None] * self.num_layers
        m_t = [None] * self.num_layers

        output = []

        # Initialize last_input with enhanced first frame features
        frame = frames[:, 0]
        last_input = self.conv_in(frame)

        # Process input sequence
        for t in range(seq_length):
            frame = frames[:, t]
            x = self.conv_in(frame)

            # Store input features for skip connection
            input_features = x

            for layer_idx in range(self.num_layers):
                h_t[layer_idx], c_t[layer_idx], m_t[layer_idx] = self.cell_list[layer_idx](
                    x, h_t[layer_idx], c_t[layer_idx], m_t[layer_idx]
                )
                x = h_t[layer_idx]

                # Add detail preservation
                if t >= seq_length - num_predictions:
                    details = self.detail_preserve[layer_idx](x)
                    x = x + details

            if t >= seq_length - num_predictions:
                # Combine features with input details
                combined = torch.cat([x, input_features], dim=1)
                output_frame = self.conv_out(combined)

                # Add brightness preservation
                mean_brightness = torch.mean(frame)  # Get input frame brightness
                output_brightness = torch.mean(output_frame)
                brightness_scale = mean_brightness / (output_brightness + 1e-6)
                output_frame = torch.clamp(output_frame * brightness_scale, 0, 1)

                output.append(output_frame)

        # Stack output frames
        output = torch.stack(output, dim=1)
        return output

    def to(self, device):
        self.device = device
        return super().to(device)
