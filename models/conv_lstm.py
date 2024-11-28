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
        h_next = o * torch.tanh(c_next) + x
        
        return h_next, c_next

class ConvLSTMPredictor(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64, kernel_size=3):
        super(ConvLSTMPredictor, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Simplified encoder - single conv layer
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # ConvLSTM layers
        self.conv_lstm1 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.conv_lstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.conv_lstm3 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        
        # Simplified decoder - single conv layer
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    @property
    def name(self):
        return "SimpleConvLSTMPredictor"

    def forward(self, x, future_frames):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize states
        h1 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        c1 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        h2 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        c2 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        h3 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        c3 = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        
        outputs = []
        
        # Process input sequence
        for t in range(seq_len):
            current_input = x[:, t]
            encoded = self.encoder(current_input)
            
            # ConvLSTM layers
            h1, c1 = self.conv_lstm1(encoded, h1, c1)
            h2, c2 = self.conv_lstm2(h1, h2, c2)
            h3, c3 = self.conv_lstm3(h2, h3, c3)
        
        # Generate future frames
        for _ in range(future_frames):
            next_frame = self.decoder(h3)
            outputs.append(next_frame)
            
            # Update states
            encoded = self.encoder(next_frame)
            h1, c1 = self.conv_lstm1(encoded, h1, c1)
            h2, c2 = self.conv_lstm2(h1, h2, c2)
            h3, c3 = self.conv_lstm3(h2, h3, c3)
        
        return torch.stack(outputs, dim=1)