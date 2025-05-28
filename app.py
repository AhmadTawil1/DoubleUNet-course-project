import streamlit as st
import zipfile
import os
import tempfile
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import gdown
import time
from model import build_doubleunet
from utils import calculate_metrics

# --------- App Setup ---------
st.set_page_config(page_title="DoubleU-Net Segmentation", layout="wide")
st.title("🦰 DoubleU-Net Medical Image Segmentation Viewer")
st.write("Upload a ZIP dataset (with `images/` and `masks/` folders) to evaluate your model.")

# --------- Download Pretrained Model ---------
checkpoint_path = "files/checkpoint.pth"
FILE_ID = "1bHytZEeCsaG_h4NPOSu4UnqUHLgwWSf0"  # ✅ Replace with your file ID
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(checkpoint_path):
    os.makedirs("files", exist_ok=True)
    with st.spinner("🔄 Downloading pretrained model..."):
        gdown.download(GDRIVE_URL, checkpoint_path, quiet=False)
    st.success("✅ Model downloaded!")

# --------- Upload ZIP Dataset ---------
uploaded_zip = st.file_uploader("📦 Upload a ZIP file with `images/` and `masks/`", type=["zip"])

if uploaded_zip:
    # Extract ZIP to a temp directory
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # 🔍 Detect images/ and masks/ folders at any depth
    image_dir = None
    mask_dir = None
    for root, dirs, files in os.walk(temp_dir):
        if 'images' in dirs and 'masks' in dirs:
            image_dir = os.path.join(root, 'images')
            mask_dir = os.path.join(root, 'masks')
            break

    if not image_dir or not mask_dir:
        st.error("❌ Could not find 'images/' and 'masks/' folders in the uploaded ZIP.")
        st.stop()

    # Load and sort image/mask files
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    # Let the user choose how many images to test
    max_images = st.slider("🔘 How many images do you want to test?", 
                           min_value=1, max_value=len(image_files), value=min(10, len(image_files)))
    image_files = image_files[:max_images]
    mask_files = mask_files[:max_images]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_doubleunet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    metrics_list = []
    start_time = time.time()
    progress = st.progress(0)

    # Inference Loop with Progress
    for idx, (img_path, mask_path) in enumerate(zip(image_files, mask_files)):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            _, y_pred = model(input_tensor)
            y_pred = torch.sigmoid(y_pred)
            score = calculate_metrics(mask_tensor, y_pred)

        metrics_list.append({
            "Image": os.path.basename(img_path),
            "Jaccard": score[0],
            "F1": score[1],
            "Recall": score[2],
            "Precision": score[3]
        })

        progress.progress((idx + 1) / len(image_files))

    end_time = time.time()
    elapsed = end_time - start_time
    st.success(f"✅ Evaluation completed in {elapsed:.2f} seconds.")

    # Show Metrics Table
    st.subheader("📊 Evaluation Metrics")
    df = pd.DataFrame(metrics_list)
    st.dataframe(df.style.format(precision=4))

    # Plot Bar Chart
    st.subheader("📈 Metric Comparison")
    metric_to_plot = st.selectbox("Select metric to visualize", ["F1", "Jaccard", "Recall", "Precision"])
    st.bar_chart(df.set_index("Image")[metric_to_plot])

    # Visual Comparison
    st.subheader("🖼️ Visual Result Viewer")
    selected_img = st.selectbox("Pick an image to view", df["Image"])
    idx = df[df["Image"] == selected_img].index[0]

    img = cv2.imread(image_files[idx])
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    gt = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)

    input_img = img.astype(np.float32) / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))
    input_tensor = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        _, y_pred = model(input_tensor)
        y_pred = torch.sigmoid(y_pred)
    pred_np = (y_pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original Image", use_column_width=True)
    col2.image(gt, caption="Ground Truth Mask", use_column_width=True)
    col3.image(pred_np, caption="Predicted Mask", use_column_width=True)
