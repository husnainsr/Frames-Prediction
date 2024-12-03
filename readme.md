# 🎥 Video Prediction using Deep Learning Models

This project implements and compares different deep learning architectures for video prediction, including ConvLSTM, PredRNN, and Transformer models. The goal is to predict future video frames based on a sequence of input frames.

## 🚀 Models Implemented

### 1. ConvLSTM
- 🧠 Convolutional LSTM for spatio-temporal prediction
- 🔄 Combines CNN and LSTM capabilities
- 📊 Suitable for capturing both spatial and temporal dependencies

### 2. PredRNN
- 🎯 Advanced architecture with Spatiotemporal LSTM cells
- 💾 Memory preservation mechanism
- 🖼️ Detail and brightness preservation features
- ⏳ Better at handling long-term dependencies

### 3. Transformer
- 👁️ Vision Transformer adaptation for video prediction
- ⚡ Positional encoding for temporal information
- 🔍 CNN-based feature extraction combined with self-attention
- 🌐 Capable of capturing global dependencies

## 🛠️ Setup Environment

***bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
***



## 📊 Data Format
- Input shape: `(batch_size, sequence_length, channels, height, width)`
- Output shape: `(batch_size, prediction_length, channels, height, width)`
- Uses H5 format for efficient data storage and loading

## 🚀 Training

```bash
python train.py --model [convlstm|predrnn|transformer] --epochs 100
```

## 📈 Testing

```bash
python test.py --model [model_name] --checkpoint [path_to_checkpoint]
```

## 🎯 Model Visualization

```bash
python visualization/model_visualizer.py --model [model_name]
```

## 📝 Citation
If you use this code for your research, please cite:
``` bibtex
@misc{video_prediction_models,
  author = {Husnain Sattar, Isma, Laiba Batool},
  title = {Video Prediction using Deep Learning Models},
  year = {2024},
  publisher = {GitHub},
  url = {[https://github.com/yourusername/repo](https://github.com/husnainsr/Frames-Generation-using-Convlstm-Transformer-and-PredRNN.git)}
}
```

## 👥 Contributors
- Husnain Sattar
- Isma
- Laiba Batool

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
