from .base_model import BaseModel
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        total_channels = input_channels + hidden_channels
        
        self.conv = nn.Conv2d(
            in_channels=total_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        
        # Split gates
        i, f, g, o = torch.split(conv_out, self.hidden_channels, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class ConvLSTMPredictor(BaseModel):
    def __init__(self, input_channels=1, hidden_channels=64, kernel_size=3):
        super(ConvLSTMPredictor, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # ConvLSTM layers
        self.conv_lstm1 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.conv_lstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    @property
    def name(self):
        return "ConvLSTMPredictor"

    def forward(self, x, future_frames):
        """
        x: Input tensor of shape (batch, time_steps, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden states
        h1 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        c1 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        h2 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        c2 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        
        outputs = []
        
        # Process input sequence
        for t in range(seq_len):
            # Get current frame and ensure correct dimension order
            current_input = x[:, t]  # (batch, channels, height, width)
            
            # Ensure the tensor is in the correct format
            if current_input.size(1) != self.input_channels:
                # Permute dimensions if necessary
                current_input = current_input.permute(0, 3, 1, 2)
            
            # Pass through encoder
            encoded_input = self.encoder_conv(current_input)
            
            # Pass through ConvLSTM layers
            h1, c1 = self.conv_lstm1(encoded_input, h1, c1)
            h2, c2 = self.conv_lstm2(h1, h2, c2)
        
        # Generate future frames
        last_frame = x[:, -1]
        if last_frame.size(1) != self.input_channels:
            last_frame = last_frame.permute(0, 3, 1, 2)
            
        for _ in range(future_frames):
            # Encode last frame
            encoded_frame = self.encoder_conv(last_frame)
            
            # Pass through ConvLSTM layers
            h1, c1 = self.conv_lstm1(encoded_frame, h1, c1)
            h2, c2 = self.conv_lstm2(h1, h2, c2)
            
            # Decode to get next frame
            next_frame = self.decoder(h2)
            
            # Ensure output is in the same format as input
            if next_frame.size(1) != channels:
                next_frame = next_frame.permute(0, 2, 3, 1)
                
            outputs.append(next_frame)
            last_frame = next_frame.permute(0, 3, 1, 2) if next_frame.size(1) != self.input_channels else next_frame
        
        # Stack outputs along time dimension
        return torch.stack(outputs, dim=1)