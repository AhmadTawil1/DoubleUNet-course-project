# Import required libraries
import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics, otsu_mask
from model import build_doubleunet
from metrics import DiceLoss, DiceBCELoss

def find_dataset_folder(base_path="datasets"):
    """Finds the first folder that contains both 'images/' and 'masks/' inside the datasets directory"""
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if os.path.exists(os.path.join(folder_path, "images")) and os.path.exists(os.path.join(folder_path, "masks")):
            print(f"✅ Detected dataset: {folder_path}")
            return folder_path
    raise Exception("❌ No valid dataset found in 'datasets/'. Please upload a folder with 'images/' and 'masks/'.")

DATASET_PATH = find_dataset_folder()

# def load_data(path):
#     """
#     Load and organize dataset into train, validation, and test sets
#     Args:
#         path: Path to the dataset directory
#     Returns:
#         Tuple of (train_data, valid_data, test_data) where each is a tuple of (images, masks)
#     """
#     def get_data(path, name):
#         """
#         Get image and mask paths for a specific dataset split
#         Args:
#             path: Base path to dataset
#             name: Name of the dataset split
#         Returns:
#             Tuple of (image_paths, mask_paths)
#         """
#         images = sorted(glob(os.path.join(path, name, "images", "*.jpg")))
#         labels = sorted(glob(os.path.join(path, name, "masks", "liver", "*.jpg")))
#         return images, labels

#     # Define dataset splits
#     dirs = sorted(os.listdir(path))
#     test_names = [f"liver_{i}" for i in range(0, 30, 1)]
#     valid_names = [f"liver_{i}" for i in range(30, 60, 1)]

#     # Get training names by excluding test and validation names
#     train_names = [item for item in dirs if item not in test_names]
#     train_names = [item for item in train_names if item not in valid_names]

#     # Load training data
#     train_x, train_y = [], []
#     for name in train_names:
#         x, y = get_data(path, name)
#         train_x += x
#         train_y += y

#     # Load validation data
#     valid_x, valid_y = [], []
#     for name in valid_names:
#         x, y = get_data(path, name)
#         valid_x += x
#         valid_y += y

#     # Load testing data
#     test_x, test_y = [], []
#     for name in test_names:
#         x, y = get_data(path, name)
#         test_x += x
#         test_y += y

#     return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

def load_data(path):
    """
    Load and split dataset into train, validation, and test sets.
    Assumes structure:
        path/
        ├── images/
        └── masks/
    Returns:
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    """
    image_paths = sorted(glob(os.path.join(path, "images", "*.jpg")))
    mask_paths = sorted(glob(os.path.join(path, "masks", "*.jpg")))

    assert len(image_paths) == len(mask_paths) and len(image_paths) > 0, \
        "❌ Dataset error: images and masks must exist and match!"

    # Shuffle the data
    combined = list(zip(image_paths, mask_paths))
    random.shuffle(combined)
    image_paths, mask_paths = zip(*combined)

    total = len(image_paths)
    train_split = int(0.8 * total)
    valid_split = int(0.9 * total)

    train_x = list(image_paths[:train_split])
    train_y = list(mask_paths[:train_split])

    valid_x = list(image_paths[train_split:valid_split])
    valid_y = list(mask_paths[train_split:valid_split])

    test_x = list(image_paths[valid_split:])
    test_y = list(mask_paths[valid_split:])

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

class DATASET(Dataset):
    """
    Custom Dataset class for loading and preprocessing images and masks
    """
    def __init__(self, images_path, masks_path, size, transform=None):
        """
        Initialize dataset
        Args:
            images_path: List of paths to input images
            masks_path: List of paths to mask images
            size: Target size for resizing images
            transform: Optional albumentations transforms for data augmentation
        """
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset
        Args:
            index: Index of the sample to get
        Returns:
            Tuple of (preprocessed_image, preprocessed_mask)
        """
        # Load image and mask
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        # Apply data augmentation if specified
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Preprocess image
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W) format
        image = image/255.0  # Normalize to [0, 1]

        # Preprocess mask
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        mask = mask/255.0  # Normalize to [0, 1]

        return image, mask

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return self.n_samples

def train(model, loader, optimizer, loss_fn, device):
    """
    Training function for one epoch
    Args:
        model: The neural network model
        loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        device: Device to run the model on (CPU/GPU)
    Returns:
        Tuple of (epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision])
    """
    model.train()

    # Initialize metrics
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x, y) in enumerate(loader):
        # Move data to device
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Forward pass
        optimizer.zero_grad()
        p1, p2 = model(x)
        loss = loss_fn(p1, y) + loss_fn(p2, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Calculate metrics for each batch
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, p2):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        # Update epoch metrics
        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    # Calculate average metrics for the epoch
    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(model, loader, loss_fn, device):
    """
    Evaluation function for one epoch
    Args:
        model: The neural network model
        loader: DataLoader for validation/test data
        loss_fn: Loss function
        device: Device to run the model on (CPU/GPU)
    Returns:
        Tuple of (epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision])
    """
    model.eval()

    # Initialize metrics
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            # Move data to device
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # Forward pass
            p1, p2 = model(x)
            loss = loss_fn(p1, y) + loss_fn(p2, y)
            epoch_loss += loss.item()

            # Calculate metrics for each batch
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, p2):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            # Update epoch metrics
            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        # Calculate average metrics for the epoch
        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__ == "__main__":
    # Set random seed for reproducibility
    seeding(42)

    # Create necessary directories
    create_dir("files")

    # Setup training log file
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    # Record training start time
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    # Define hyperparameters
    image_size = 256
    size = (image_size, image_size)
    batch_size = 16
    num_epochs = 30
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = "files/checkpoint.pth"
    path = DATASET_PATH

    # Log hyperparameters
    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    # Load and prepare dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)

    # Limit dataset size for faster training (optional)
    # train_x = train_x[:500]
    # train_y = train_y[:500]
    # valid_x = valid_x[:500]
    # valid_y = valid_y[:500]

    # Log dataset sizes
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    # Define data augmentation transforms
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    # Create datasets and dataloaders
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = build_doubleunet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
