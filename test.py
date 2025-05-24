# Import required libraries
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from model import build_doubleunet
from utils import create_dir, seeding
from utils import calculate_metrics, otsu_mask
from train import load_data

def process_mask(y_pred):
    """
    Process the predicted mask for visualization
    Args:
        y_pred: Raw prediction from the model
    Returns:
        Processed mask as a 3-channel image
    """
    # Convert tensor to numpy array
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    
    # Convert to binary mask
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    
    # Convert to 3-channel image for visualization
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred

def print_score(metrics_score):
    """
    Print evaluation metrics averaged over the test set
    Args:
        metrics_score: List of accumulated metric scores
    """
    # Calculate average scores
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)

    # Print formatted scores
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

def evaluate(model, save_path, test_x, test_y, size):
    """
    Evaluate the model on test data and save results
    Args:
        model: Trained model
        save_path: Path to save results
        test_x: List of test image paths
        test_y: List of test mask paths
        size: Target size for resizing images
    """
    # Initialize metric scores for both U-Nets
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    metrics_score_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        # Get image name for saving results
        name = x.split("/")
        name = f"{name[-3]}_{name[-1]}"

        # Load and preprocess input image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image  # Keep original for visualization
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image/255.0  # Normalize
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        # Load and preprocess ground truth mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask  # Keep original for visualization
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            # Measure inference time
            start_time = time.time()
            y_pred1, y_pred2 = model(image)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            # Apply sigmoid to get probabilities
            y_pred1 = torch.sigmoid(y_pred1)
            y_pred2 = torch.sigmoid(y_pred2)

            # Calculate metrics for first U-Net
            score_1 = calculate_metrics(mask, y_pred1)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))

            # Calculate metrics for second U-Net
            score_2 = calculate_metrics(mask, y_pred2)
            metrics_score_2 = list(map(add, metrics_score_2, score_2))

            # Process predictions for visualization
            y_pred1 = process_mask(y_pred1)
            y_pred2 = process_mask(y_pred2)

        # Create visualization by concatenating images
        line = np.ones((size[0], 10, 3)) * 255  # White separator line
        cat_images = np.concatenate([save_img, line, save_mask, line, y_pred1, line, y_pred2], axis=1)
        
        # Save results
        cv2.imwrite(f"{save_path}/joint/{name}", cat_images)
        cv2.imwrite(f"{save_path}/mask1/{name}", y_pred1)
        cv2.imwrite(f"{save_path}/mask2/{name}", y_pred2)

    # Print evaluation metrics
    print_score(metrics_score_1)
    print_score(metrics_score_2)

    # Calculate and print average FPS
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)


if __name__ == "__main__":
    # Set random seed for reproducibility
    seeding(42)

    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_doubleunet()
    model = model.to(device)
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load test dataset
    path = "../../Task03_Liver"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    # Create directories for saving results
    save_path = f"results"
    for item in ["mask1", "mask2", "joint"]:
        create_dir(f"{save_path}/{item}")

    # Set image size and run evaluation
    size = (256, 256)
    evaluate(model, save_path, test_x, test_y, size)
