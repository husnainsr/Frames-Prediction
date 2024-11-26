from models.conv_lstm import ConvLSTMPredictor
from training.trainer import ModelTrainer, VideoDataset
from torch.utils.data import DataLoader

def main():
    # Training configuration
    config = {
        'batch_size': 12,
        'learning_rate': 0.0001,
        'epochs': 50,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'log_interval': 50,
        'input_frames': 10,
        'output_frames': 5,
        'checkpoint_interval': 5,
        'augmentation': False
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
        hidden_channels=128,
        kernel_size=3
    )
    
    trainer = ModelTrainer(model, config)
    
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()