import sys
from pathlib import Path
import os
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from models.predrnn import PredRNN
from models.conv_lstm import ConvLSTMPredictor
from models.transformer import TransformerModel
from torchvision.transforms import ToTensor, ToPILImage
import base64

def load_model(model_name):
    if model_name == "PredRNN":
        model = PredRNN(
            input_channels=1,
            hidden_channels=32,
            num_layers=3,
            num_frames_output=5
        )
    elif model_name == "ConvLSTM":
        model = ConvLSTMPredictor(
            input_channels=1,
            hidden_channels=64,
            kernel_size=3
        )
    elif model_name == "Transformer":
        model = TransformerModel(
            input_dim=64*64,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
            output_dim=64*64,
            seq_length=10
        )
    else:
        raise ValueError("Unknown model name")

    checkpoint = torch.load(f'models/{model_name}_best.pth', map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    return np.array(frames)

def create_gif(frames, path, fps=10):
    pil_images = [Image.fromarray(frame) for frame in frames]
    pil_images[0].save(
        path,
        save_all=True,
        append_images=pil_images[1:],
        duration=1000 // fps,
        loop=0
    )

def encode_gif(gif_path):
    with open(gif_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def process_video(uploaded_file, device):
    video_path = f"temp_video.{uploaded_file.name.split('.')[-1]}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    frames = extract_frames(video_path)
    frames = [cv2.resize(frame, (64, 64)) for frame in frames]
    
    input_frames = torch.stack([
        ToTensor()(Image.fromarray(frame)) for frame in frames
    ]).unsqueeze(0).to(device)

    if input_frames.size(1) > 10:
        input_frames = input_frames[:, :10]
    elif input_frames.size(1) < 10:
        pad_size = 10 - input_frames.size(1)
        input_frames = torch.cat([
            input_frames,
            torch.zeros(1, pad_size, 1, 64, 64, device=device)
        ], dim=1)
    
    return frames, input_frames

def single_model_view():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_name = st.selectbox(
            "🤖 Select Model Architecture",
            ["PredRNN", "ConvLSTM", "Transformer"],
            help="Choose the deep learning model for frame prediction"
        )

    with col2:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.markdown(
            f"""
            <div class='device-info'>
                💻 Device: {device}<br>
                🔧 Status: {'GPU Enabled' if device.type == 'cuda' else 'CPU Mode'}
            </div>
            """,
            unsafe_allow_html=True
        )

    uploaded_file = st.file_uploader(
        "Choose a video file (max 5MB)",
        type=["mp4", "avi", "mov"],
        help="Upload a video file to predict future frames (maximum size: 5MB)",
        key="single_model_uploader",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Check file size (5MB = 5 * 1024 * 1024 bytes)
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("Please upload a video file smaller than 5MB")
            return

        model = load_model(model_name).to(device)
        frames, input_frames = process_video(uploaded_file, device)

        with torch.no_grad():
            if model_name == "Transformer":
                predicted_frames = model(input_frames).squeeze(0).cpu()
            elif model_name == "PredRNN":
                predicted_frames = model(input_frames, num_predictions=5).squeeze(0).cpu()
            elif model_name == "ConvLSTM":
                predicted_frames = model(input_frames, future_frames=5).squeeze(0).cpu()

        # Create a directory to save the frames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"output/{model_name}_{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save input frames
        input_images = [Image.fromarray(frame) for frame in frames[:10]]
        for i, img in enumerate(input_images):
            img.save(save_dir / f"input_frame_{i+1}.png")

        # Save predicted frames
        predicted_images = [ToPILImage()(frame.squeeze()) for frame in predicted_frames]
        for i, img in enumerate(predicted_images):
            img.save(save_dir / f"predicted_frame_{i+1}.png")

        # Save target frames (assuming target frames are the last 5 frames of the input)
        target_images = [Image.fromarray(frame) for frame in frames[-5:]]
        for i, img in enumerate(target_images):
            img.save(save_dir / f"target_frame_{i+1}.png")

        st.markdown("<div class='subheader'>📥 Input Sequence</div>", unsafe_allow_html=True)
        cols = st.columns(10)
        for i, col in enumerate(cols):
            col.image(input_images[i], caption=f"Frame {i+1}", use_container_width=True)

        st.markdown("<div class='subheader'>🔮 Predicted Sequence</div>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, col in enumerate(cols):
            col.image(predicted_images[i], caption=f"Predicted {i+1}", use_container_width=True)

        st.markdown("<div class='subheader'>🎬 Animation</div>", unsafe_allow_html=True)
        gif_path = "predicted_frames.gif"
        create_gif([np.array(img) for img in predicted_images], gif_path)
        
        st.markdown(
            f"""
            <div class='gif-container'>
                <img src='data:image/gif;base64,{encode_gif(gif_path)}' width='300px'>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(f"Frames saved to {save_dir}")

def model_comparison_view():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.markdown(
        f"""
        <div class='device-info'>
            💻 Device: {device}<br>
            🔧 Status: {'GPU Enabled' if device.type == 'cuda' else 'CPU Mode'}
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a video file (max 5MB)",
        type=["mp4", "avi", "mov"],
        help="Upload a video file to compare predictions across models (maximum size: 5MB)",
        key="comparison_uploader",
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        frames, input_frames = process_video(uploaded_file, device)
        
        col1, col2, col3 = st.columns(3)
        models = ["PredRNN", "ConvLSTM", "Transformer"]
        cols = [col1, col2, col3]

        for model_name, col in zip(models, cols):
            with col:
                st.markdown(f"<h3 style='text-align: center;'>{model_name}</h3>", unsafe_allow_html=True)
                
                model = load_model(model_name).to(device)
                
                with torch.no_grad():
                    if model_name == "Transformer":
                        predicted_frames = model(input_frames).squeeze(0).cpu()
                    elif model_name == "PredRNN":
                        predicted_frames = model(input_frames, num_predictions=5).squeeze(0).cpu()
                    elif model_name == "ConvLSTM":
                        predicted_frames = model(input_frames, future_frames=5).squeeze(0).cpu()

                predicted_images = [ToPILImage()(frame.squeeze()) for frame in predicted_frames]
                gif_path = f"predicted_frames_{model_name.lower()}.gif"
                create_gif([np.array(img) for img in predicted_images], gif_path)
                
                st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <img src='data:image/gif;base64,{encode_gif(gif_path)}' width='250px'>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def main():
    st.set_page_config(
        page_title="Video Frame Prediction",
        page_icon="🎥",
        layout="wide"
    )

    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            padding-bottom: 2rem;
        }
        .stSelectbox {
            margin-bottom: 2rem;
        }
        .subheader {
            color: #34495e;
            padding: 1rem 0;
            border-bottom: 2px solid #eee;
            margin-bottom: 1rem;
        }
        .gif-container {
            display: flex;
            justify-content: center;
            padding: 2rem 0;
        }
        .status-container {
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .device-info {
            padding: 0.5rem 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>🎥 Video Frame Prediction</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Single Model", "Model Comparison"])

    with tab1:
        single_model_view()

    with tab2:
        model_comparison_view()

    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            Made with ❤️ 
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()