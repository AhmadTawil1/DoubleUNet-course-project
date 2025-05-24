# Import required libraries
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from metrics import precision, recall, F2, dice_score, jac_score
from sklearn.metrics import accuracy_score, confusion_matrix

def seeding(seed):
    """
    Set random seeds for reproducibility across different libraries
    Args:
        seed: Integer value to set as the random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    """
    Create a directory if it doesn't exist
    Args:
        path: Path of the directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    """
    Shuffle the input data and labels while maintaining their correspondence
    Args:
        x: Input data array
        y: Target labels array
    Returns:
        Shuffled x and y arrays
    """
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    """
    Calculate the elapsed time between start and end times
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
    Returns:
        Tuple of (elapsed minutes, elapsed seconds)
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    """
    Print data to console and append it to a file
    Args:
        file_path: Path to the file where data should be saved
        data_str: String data to print and save
    """
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def otsu_mask(image, size):
    """
    Create a binary mask using Otsu's thresholding method
    Args:
        image: Path to the input image
        size: Target size for resizing the image (width, height)
    Returns:
        Binary mask as numpy array
    """
    # Read image in grayscale
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # Resize image to target size
    img = cv2.resize(img, size)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(img,(5,5),0)
    # Apply Otsu's thresholding
    ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Convert to binary mask
    th = th.astype(np.int32)
    th = th/255.0
    th = th > 0.5
    th = th.astype(np.int32)
    return th

def calculate_metrics(y_true, y_pred):
    """
    Calculate various evaluation metrics for binary segmentation
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    Returns:
        List of metrics [Jaccard, Dice, Recall, Precision, Accuracy, F2]
    """
    # Convert tensors to numpy arrays
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Convert predictions to binary
    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    # Convert ground truth to binary
    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    # Calculate various metrics
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]
