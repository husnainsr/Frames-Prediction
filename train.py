from models.conv_lstm import ConvLSTMPredictor
from training.trainer import ModelTrainer, VideoDataset
from torch.utils.data import DataLoader

def main():
    # Training configuration
    config = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 20,
        'log_interval': 10,
        'input_frames': 10,
        'output_frames': 5,
        'checkpoint_interval': 5  # Save checkpoint every 5 epochs
    }
    
    # Create datasets
    train_dataset = VideoDataset('./processed_data/train_sequences.h5')
    val_dataset = VideoDataset('./processed_data/val_sequences.h5')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = ConvLSTMPredictor(
        input_channels=1,
        hidden_channels=32,
        kernel_size=3
    )
    
    trainer = ModelTrainer(model, config)
    
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()