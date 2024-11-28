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
    def __init__(self, input_channels=1, hidden_channels=128, kernel_size=3):
        super(ConvLSTMPredictor, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Encoder layers
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        
        # ConvLSTM layers
        self.conv_lstm1 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.conv_lstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.conv_lstm3 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        
        # Decoder layers
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, kernel_size=4, stride=2, padding=1),  # +32 for skip connection
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure output is in [0,1] range
        )
        
        # Attention and motion encoder
        self.attention = nn.MultiheadAttention(hidden_channels, 8)
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2)
        )

    @property
    def name(self):
        return "ConvLSTMPredictor"

    def forward(self, x, future_frames):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize states for all LSTM layers
        h1 = torch.zeros(batch_size, self.hidden_channels, height//4, width//4, device=x.device)
        c1 = torch.zeros(batch_size, self.hidden_channels, height//4, width//4, device=x.device)
        h2 = torch.zeros(batch_size, self.hidden_channels, height//4, width//4, device=x.device)
        c2 = torch.zeros(batch_size, self.hidden_channels, height//4, width//4, device=x.device)
        h3 = torch.zeros(batch_size, self.hidden_channels, height//4, width//4, device=x.device)
        c3 = torch.zeros(batch_size, self.hidden_channels, height//4, width//4, device=x.device)
        
        outputs = []
        skip_features = []
        prev_h = None
        
        # Process input sequence
        for t in range(seq_len):
            current_input = x[:, t]
            if current_input.size(1) != self.input_channels:
                current_input = current_input.permute(0, 3, 1, 2)
            
            # Encode
            skip = self.encoder_conv1(current_input)
            encoded = self.encoder_conv2(skip)
            skip_features.append(skip)
            
            # ConvLSTM layers
            h1, c1 = self.conv_lstm1(encoded, h1, c1)
            h2, c2 = self.conv_lstm2(h1, h2, c2)
            h3, c3 = self.conv_lstm3(h2, h3, c3)
            
            # Store motion information
            if prev_h is not None:
                motion = self.motion_encoder(torch.cat([h3, prev_h], dim=1))
                h3 = h3 + motion
            prev_h = h3.clone()
        
        # Generate future frames
        for _ in range(future_frames):
            # Decode with motion-aware features
            decoded = self.decoder_conv1(h3)
            decoded = torch.cat([decoded, skip_features[-1]], dim=1)
            next_frame = self.decoder_conv2(decoded)
            
            outputs.append(next_frame)
            
            # Update hidden states with motion information
            encoded = self.encoder_conv2(self.encoder_conv1(next_frame))
            h1, c1 = self.conv_lstm1(encoded, h1, c1)
            h2, c2 = self.conv_lstm2(h1, h2, c2)
            h3, c3 = self.conv_lstm3(h2, h3, c3)
            
            if prev_h is not None:
                motion = self.motion_encoder(torch.cat([h3, prev_h], dim=1))
                h3 = h3 + motion
            prev_h = h3.clone()
        
        return torch.stack(outputs, dim=1)