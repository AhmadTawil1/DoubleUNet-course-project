import torch
import torch.nn as nn
import torch.nn.functional as F

""" Loss Functions -------------------------------------- """
class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Measures overlap between predicted and ground truth masks
    Args:
        weight: Optional weight parameter (not used in this implementation)
        size_average: Whether to average the loss (not used in this implementation)
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass to calculate Dice Loss
        Args:
            inputs: Predicted probabilities
            targets: Ground truth labels
            smooth: Smoothing factor to prevent division by zero
        Returns:
            Dice Loss value (1 - Dice coefficient)
        """
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy Loss
    Combines the benefits of both loss functions for better training
    Args:
        weight: Optional weight parameter (not used in this implementation)
        size_average: Whether to average the loss (not used in this implementation)
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass to calculate combined Dice and BCE Loss
        Args:
            inputs: Predicted probabilities
            targets: Ground truth labels
            smooth: Smoothing factor to prevent division by zero
        Returns:
            Combined Dice and BCE Loss value
        """
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    """
    Calculate precision score for binary segmentation
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    Returns:
        Precision score (true positives / (true positives + false positives))
    """
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    """
    Calculate recall score for binary segmentation
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    Returns:
        Recall score (true positives / (true positives + false negatives))
    """
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    """
    Calculate F-beta score (F2 score) for binary segmentation
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        beta: Beta parameter (default=2, giving more weight to recall)
    Returns:
        F2 score
    """
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    """
    Calculate Dice coefficient for binary segmentation
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    Returns:
        Dice coefficient (2 * intersection / (sum of both sets))
    """
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    """
    Calculate Jaccard score (Intersection over Union) for binary segmentation
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    Returns:
        Jaccard score (intersection / union)
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)
