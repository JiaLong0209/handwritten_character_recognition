import os
import zipfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import numpy as np
import pandas as pd
from tqdm import tqdm  # 添加進度條庫
import torch.amp as amp
from torch.cuda.amp import GradScaler
import json

# Create training results directory
training_results_dir = os.path.join(os.getcwd(), 'training_results')
os.makedirs(training_results_dir, exist_ok=True)

pth_dir = os.path.join(os.getcwd(), 'pth')
os.makedirs(pth_dir, exist_ok=True)


# Step 1: Setup parameters and check CUDA
def check_cuda():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA available, using GPU training")
        print(f"Current GPU name: {device_name}")
        # print(f"GPU count: {device_count}")
        # print(f"Currently using GPU: {device_name}")
        # print(f"CUDA version: {torch.version.cuda}")
        # print(f"PyTorch version: {torch.__version__}")
        return True
    else:
        print("CUDA not available, using CPU training")
        return False

use_gpu = check_cuda()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")

# Set local path
data_dir = os.path.join(os.getcwd(), 'Handwritten_Data')  # Use absolute path
input_size = 224

# Set CUDA optimization parameters
if use_gpu:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set number of workers based on CPU cores, but limit to avoid memory issues
    num_worker = min(4, os.cpu_count() or 1)  # Reduced to 4 workers for stability
    # Set memory fraction to use (adjust based on your GPU memory)
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
else:
    num_worker = 0

# Global variables for data loading
train_loader_global = None
val_loader_global = None
train_size_global = None
val_size_global = None
class_num_global = None
class_names_global = None

# Add after imports, before any functions
# Training configuration
USE_OPTUNA = False  # Set to True to use Optuna optimization, False for direct training
# DEFAULT_PARAMS = {
#     'lr': 0.001,
#     'batch_size': 128,
#     'optimizer': 'Adam',
#     'weight_decay': 1e-4,
#     'scheduler': 'StepLR',
#     'gamma': 0.7,
#     'dropout_rate': 0.3,
#     'step_size': 2
# }

DEFAULT_PARAMS = {
    'lr': 0.0001329291894316216,
    'batch_size': 32,
    'optimizer': 'Adam',
    'weight_decay': 0.0003967605077052988,
    'scheduler': 'ExponentialLR',
    'gamma': 0.9364594334728974,
    'dropout_rate': 0.4329770563201687
}

subsample_rate = 1
# subsample_rate = 0.10

class DatasetWrapper(torch.utils.data.Dataset):
    """Wrapper class for dataset with custom transforms"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, label = self.subset.dataset[self.subset.indices[idx]]
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# Step 2: Define data loading functions

from sklearn.model_selection import train_test_split

def loaddata(data_dir, batch_size, shuffle=True, subsample_rate=1.0, subsample_val=True):
    """Using stratified sampling for better class balance control"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    raw_dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())
    
    # Get all indices and labels more efficiently
    all_indices = torch.arange(len(raw_dataset))
    all_labels = torch.tensor(raw_dataset.targets)

    # Stratified train-val split
    train_indices, val_indices = train_test_split(
        all_indices.numpy(),
        test_size=0.2,
        stratify=all_labels.numpy(),
        random_state=42
    )
    
    # Apply subsampling to training set
    if subsample_rate < 1.0:
        train_labels = all_labels[train_indices]
        subsampled_train_indices, _ = train_test_split(
            train_indices,
            train_size=subsample_rate,
            stratify=train_labels.numpy(),
            random_state=42
        )
        train_indices = subsampled_train_indices
    
    # Apply subsampling to validation set if specified
    if subsample_val and subsample_rate < 1.0:
        val_labels = all_labels[val_indices]
        subsampled_val_indices, _ = train_test_split(
            val_indices,
            train_size=subsample_rate,
            stratify=val_labels.numpy(),
            random_state=42
        )
        val_indices = subsampled_val_indices
    
    train_subset = torch.utils.data.Subset(raw_dataset, train_indices)
    val_subset = torch.utils.data.Subset(raw_dataset, val_indices)
    
    # Use the global DatasetWrapper class
    train_dataset_wrapped = DatasetWrapper(train_subset, data_transforms['train'])
    val_dataset_wrapped = DatasetWrapper(val_subset, data_transforms['val'])
    
    # Enable multithreading with proper settings
    train_loader = DataLoader(
        train_dataset_wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        pin_memory=use_gpu,
        persistent_workers=num_worker > 0,
        prefetch_factor=2 if num_worker > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=use_gpu,
        persistent_workers=num_worker > 0,
        prefetch_factor=2 if num_worker > 0 else None
    )
    
    print(f"Data loading complete - Train: {len(train_indices)}, Val: {len(val_indices)}, Classes: {len(raw_dataset.classes)}")
    return train_loader, val_loader, len(train_indices), len(val_indices), len(raw_dataset.classes), raw_dataset.classes

# Learning rate scheduler
def get_lr_scheduler(optimizer, scheduler_type='StepLR', step_size=2, gamma=0.5):
    """Create learning rate scheduler"""
    if scheduler_type == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training function for Optuna optimization
def train_model_optuna(model, criterion, optimizer, scheduler, train_loader, val_loader, train_size, val_size, num_epochs=3):
    """Optimized training function with CUDA optimizations"""
    best_acc = 0.0
    # Fix deprecation warning
    scaler = torch.amp.GradScaler('cuda')  # Updated to new format
    
    for epoch in range(num_epochs):
        print(f"\n  Optuna Trial - Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_pbar = tqdm(train_loader, desc="Training", ncols=100)
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Fix deprecation warning
            with torch.amp.autocast('cuda'):  # Updated to new format
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if train_pbar.n % 100 == 0:
                current_acc = running_corrects.double() / (train_pbar.n * inputs.size(0))
                train_pbar.set_postfix(loss=f'{loss.item():.3f}', acc=f'{current_acc:.3f}')
        
        train_pbar.close()
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        val_pbar = tqdm(val_loader, desc="Validation", ncols=100)
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # Fix deprecation warning
                with torch.amp.autocast('cuda'):  # Updated to new format
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                
                if val_pbar.n % 50 == 0:
                    val_pbar.set_postfix(loss=f'{loss.item():.3f}')
        
        val_pbar.close()
        
        epoch_acc = val_running_corrects.double() / val_size
        epoch_loss = val_running_loss / val_size
        
        print(f"  Epoch {epoch+1} - Train Acc: {running_corrects.double() / train_size:.4f}, Val Acc: {epoch_acc:.4f}, Val Loss: {epoch_loss:.4f}")
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
        
        scheduler.step()
        
        # Clear GPU cache periodically
        if use_gpu:
            torch.cuda.empty_cache()
    
    return best_acc.item()

# Training function with full metrics
def train_model_full(model, criterion, optimizer, scheduler, train_loader, val_loader, train_size, val_size, num_epochs=5):
    """Complete training function with CUDA optimizations"""
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Fix deprecation warning
    scaler = torch.amp.GradScaler('cuda')  # Updated to new format
    
    # Recording lists
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    
    print(f"Starting training with mixed precision on {device}")
    print("=" * 50)
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Full Training", unit="epoch")
    
    for epoch in epoch_pbar:
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_train_preds = []
        all_train_labels = []
        
        # Training batch progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False, unit="batch")
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Use optimizer.zero_grad(set_to_none=True) for efficiency
            optimizer.zero_grad(set_to_none=True)
            
            # Use mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_train_preds.append(preds.cpu())
            all_train_labels.append(labels.cpu())
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(running_corrects.double() / ((train_pbar.n + 1) * inputs.size(0))):.4f}'
            })
        
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu().item())
        
        # Calculate training Precision/Recall/F1
        train_preds = torch.cat(all_train_preds)
        train_labels = torch.cat(all_train_labels)
        train_precisions.append(precision_score(train_labels, train_preds, average='macro', zero_division=0))
        train_recalls.append(recall_score(train_labels, train_preds, average='macro', zero_division=0))
        train_f1s.append(f1_score(train_labels, train_preds, average='macro', zero_division=0))
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_val_preds = []
        all_val_labels = []
        
        # Validation batch progress bar
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False, unit="batch")
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # Use mixed precision training for validation
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.cpu().item())
        
        # Calculate validation Precision/Recall/F1
        val_preds = torch.cat(all_val_preds)
        val_labels = torch.cat(all_val_labels)
        val_precisions.append(precision_score(val_labels, val_preds, average='macro', zero_division=0))
        val_recalls.append(recall_score(val_labels, val_preds, average='macro', zero_division=0))
        val_f1s.append(f1_score(val_labels, val_preds, average='macro', zero_division=0))
        
        scheduler.step()
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict().copy()
            
            # Save checkpoint in training_results directory
            checkpoint_path = os.path.join(pth_dir, 'best_model_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': epoch_loss,
                'acc': epoch_acc,
            }, checkpoint_path)
            print(f'New best model saved, accuracy: {best_acc:.4f}')
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'best_acc': f'{best_acc:.4f}',
            'current_acc': f'{epoch_acc:.4f}',
            'loss': f'{epoch_loss:.4f}'
        })
        
        # Clear GPU cache periodically
        if use_gpu:
            torch.cuda.empty_cache()
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    
    # Save model in training_results directory
    model_path = os.path.join(pth_dir, 'best_model.pth')
    torch.save(model, model_path)
    print(f'Final model saved to: {model_path}')
    
    # Plot training curves in training_results directory
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        train_precisions, val_precisions, train_recalls, val_recalls,
                        train_f1s, val_f1s, num_epochs, training_results_dir)
    
    return model, best_acc

def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        train_precisions, val_precisions, train_recalls, val_recalls,
                        train_f1s, val_f1s, num_epochs, save_dir):
    """Plot training curves (save only, no display)"""
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accs, 'b-', label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs+1), train_precisions, 'b--', label='Train Precision')
    plt.plot(range(1, num_epochs+1), val_precisions, 'r--', label='Val Precision')
    plt.plot(range(1, num_epochs+1), train_recalls, 'b:', label='Train Recall')
    plt.plot(range(1, num_epochs+1), val_recalls, 'r:', label='Val Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(range(1, num_epochs+1), train_f1s, 'b-', label='Train F1-Score')
    plt.plot(range(1, num_epochs+1), val_f1s, 'r-', label='Val F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'training_curves_all_metrics.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure without displaying
    print(f'Training curves saved to: {curve_path}')

def plot_optuna_results(study):
    """Plot Optuna optimization results (save only, no display)"""
    print("\nSaving Optuna optimization results...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Optimization history
    try:
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0,0])
        axes[0,0].set_title('Optimization History')
    except Exception as e:
        print(f"Cannot plot optimization history: {e}")
        axes[0,0].text(0.5, 0.5, 'Optimization History\nNot Available', ha='center', va='center')
    
    # 2. Parameter importance
    try:
        optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0,1])
        axes[0,1].set_title('Parameter Importance')
    except Exception as e:
        print(f"Cannot plot parameter importance: {e}")
        axes[0,1].text(0.5, 0.5, 'Parameter Importance\nNot Available', ha='center', va='center')
    
    # 3. Trial value distribution
    values = [trial.value for trial in study.trials if trial.value is not None]
    if values:
        axes[1,0].hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].set_title('Trial Value Distribution')
        axes[1,0].set_xlabel('Accuracy')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Best parameters table
    axes[1,1].axis('off')
    best_params = study.best_params
    best_value = study.best_value
    
    param_text = f"Best Accuracy: {best_value:.4f}\n\nBest Parameters:\n"
    for key, value in best_params.items():
        param_text += f"{key}: {value}\n"
    
    axes[1,1].text(0.1, 0.9, param_text, transform=axes[1,1].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1,1].set_title('Best Parameters')
    
    plt.tight_layout()
    
    # Save chart in training_results directory
    optuna_results_path = os.path.join(training_results_dir, 'optuna_optimization_results.png')
    plt.savefig(optuna_results_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure without displaying
    print(f'Optuna optimization results saved to: {optuna_results_path}')
    
    # Create detailed parameter comparison chart
    plot_parameter_comparison(study)

def plot_parameter_comparison(study):
    """Plot parameter comparison chart (save only, no display)"""
    print("Saving parameter comparison chart...")
    
    df = study.trials_dataframe()
    if len(df) == 0:
        print("No trial data available for plotting")
        return
    
    # Extract numeric parameters
    numeric_params = []
    for col in df.columns:
        if col.startswith('params_') and df[col].dtype in ['float64', 'int64']:
            numeric_params.append(col)
    
    if len(numeric_params) >= 2:
        n_params = len(numeric_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, param in enumerate(numeric_params):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[col]
            
            # Scatter plot: parameter value vs accuracy
            x = df[param].values
            y = df['value'].values
            
            # Remove NaN values
            mask = ~(pd.isna(x) | pd.isna(y))
            x = x[mask]
            y = y[mask]
            
            if len(x) > 0:
                ax.scatter(x, y, alpha=0.6)
                ax.set_xlabel(param.replace('params_', ''))
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{param.replace("params_", "")} vs Accuracy')
                ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row][col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        param_comp_path = os.path.join(training_results_dir, 'parameter_comparison.png')
        plt.savefig(param_comp_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure without displaying
        print(f'Parameter comparison chart saved to: {param_comp_path}')

def test_samples(model, val_loader, class_names, device, num_samples=10):
    """Test several samples"""
    model.eval()
    samples_tested = 0
    correct_predictions = 0
    
    print(f"\nTesting {num_samples} samples...")
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(min(len(preds), num_samples - samples_tested)):
                pred_class = class_names[preds[i]]
                true_class = class_names[labels[i]]
                is_correct = preds[i] == labels[i]
                
                status = "✓" if is_correct else "✗"
                print(f"{status} Predicted: {pred_class} | Actual: {true_class}")
                
                if is_correct:
                    correct_predictions += 1
                samples_tested += 1
                
                if samples_tested >= num_samples:
                    break
            
            if samples_tested >= num_samples:
                break
    
    accuracy = correct_predictions / samples_tested
    print(f"\nTest sample accuracy: {accuracy:.4f} ({correct_predictions}/{samples_tested})")


def main():
    """Main program"""

    num_epochs_final = 10  # Reduced to 5 epochs for final training

    print("Starting handwritten character recognition training")
    print("=" * 70)
    if USE_OPTUNA:
        print("Using Optuna for hyperparameter optimization")
    else:
        print("Using default parameters for training")
    print("=" * 70)
    
    # Check data path
    if not os.path.exists(data_dir):
        print(f"Data folder does not exist: {data_dir}")
        print("Please confirm if the data folder path is correct")
        return
    
    print(f"Using data directory: {data_dir}")
    print("Loading data...")
    try:
        # Load data
        batch_size = DEFAULT_PARAMS['batch_size'] if not USE_OPTUNA else 64  # Use default or temporary for Optuna
        train_loader, val_loader, train_size, val_size, class_num, class_names = loaddata(
            data_dir, 
            batch_size=batch_size, 
            subsample_rate=subsample_rate
        )
        print(f"Data loaded successfully")
        print(f"   Training samples: {train_size}")
        print(f"   Validation samples: {val_size}")
        print(f"   Character classes: {class_num}")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        print("Please check if:")
        print("  1. The data directory contains image files")
        print("  2. The images are in supported formats (.jpg, .jpeg, .png, etc.)")
        print("  3. The directory structure is correct (each class in its own folder)")
        return

    if USE_OPTUNA:
        # === Optuna hyperparameter optimization ===
        print(f"\nStarting Optuna hyperparameter optimization...")
        print("This will perform multiple trials to find the best parameter combination")
        
        # Create study
        study = optuna.create_study(direction='maximize', 
                                  study_name='handwriting_optimization',
                                  sampler=optuna.samplers.TPESampler(seed=42))
        
        # Execute optimization with simple progress tracking
        n_trials = 5
        print(f"Will perform {n_trials} trials...")
        
        for trial_num in range(n_trials):
            print(f"\n{'='*60}")
            print(f"🔍 OPTUNA TRIAL {trial_num + 1}/{n_trials}")
            print(f"{'='*60}")
            
            try:
                study.optimize(objective, n_trials=1, timeout=None)
                
                if study.best_value:
                    print(f"✅ Trial {trial_num + 1} completed!")
                    print(f"📊 Current Best Accuracy: {study.best_value:.4f}")
                    print(f"🎯 Best Parameters so far: {study.best_params}")
                else:
                    print(f"❌ Trial {trial_num + 1} failed to complete")
                    
            except KeyboardInterrupt:
                print(f"\n⏹️  Trial {trial_num + 1} interrupted by user")
                break
            except Exception as e:
                print(f"❌ Trial {trial_num + 1} failed with error: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"🏁 OPTUNA OPTIMIZATION COMPLETED")
        print(f"{'='*60}")
        
        # Show best results
        print(f"\nOptuna optimization complete!")
        print(f"Best accuracy: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        # Plot optimization results
        plot_optuna_results(study)
        
        # Use best parameters for training
        best_params = study.best_params
    else:
        # Use default parameters
        best_params = DEFAULT_PARAMS
        print("\nUsing default parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    
    # === Training with selected parameters ===
    print(f"\nStarting training with {'optimized' if USE_OPTUNA else 'default'} parameters...")
    
    # Reload data with final batch size
    train_loader, val_loader, train_size, val_size, class_num, class_names = loaddata(
        data_dir, best_params['batch_size'], subsample_rate=subsample_rate)
    
    # Build model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(best_params['dropout_rate']),
        nn.Linear(num_ftrs, class_num)
    )
    model = model.to(device)
    
    # Set optimizer
    if best_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                              lr=best_params['lr'], 
                              weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), 
                               lr=best_params['lr'], 
                               weight_decay=best_params['weight_decay'])
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), 
                             lr=best_params['lr'], 
                             momentum=0.9, 
                             weight_decay=best_params['weight_decay'])
    
    # Set learning rate scheduler
    if best_params['scheduler'] == 'StepLR':
        scheduler = get_lr_scheduler(optimizer, 'StepLR', 
                                   best_params.get('step_size', 2), 
                                   best_params['gamma'])
    else:
        scheduler = get_lr_scheduler(optimizer, best_params['scheduler'], 
                                   gamma=best_params['gamma'])
    
    criterion = nn.CrossEntropyLoss()
    
    # Complete training
    try:
        model, best_acc = train_model_full(
            model, criterion, optimizer, scheduler,
            train_loader, val_loader,
            train_size, val_size,
            num_epochs=num_epochs_final
        )
        
        print(f"\n🎉 Complete training finished! Final accuracy: {best_acc:.4f}")
        
        # Test several samples
        # print("\nTesting several samples...")
        # test_samples(model, val_loader, class_names, device)
        
        # Save best parameters to file
        params_file = os.path.join(training_results_dir, 'best_hyperparameters.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print(f"Best hyperparameters saved to: {params_file}")
        
        if USE_OPTUNA:
            # Create optimization summary report
            create_optimization_summary(study, best_acc, training_results_dir)
        
    except Exception as e:
        print(f"Error occurred during complete training: {e}")
        return

def create_optimization_summary(study, final_accuracy, save_dir):
    """Create optimization summary report (all in English)"""
    print("\nCreating optimization summary report...")
    
    # Collect statistical information
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    if len(completed_trials) == 0:
        print("No completed trials to analyze")
        return
    
    summary = {
        'optimization_summary': {
            'total_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'best_trial_number': study.best_trial.number,
            'best_validation_accuracy': study.best_value,
            'final_training_accuracy': final_accuracy,
            'improvement': final_accuracy - study.best_value if study.best_value else 0
        },
        'best_parameters': study.best_params,
        'statistics': {
            'mean_accuracy': float(completed_trials['value'].mean()),
            'std_accuracy': float(completed_trials['value'].std()),
            'min_accuracy': float(completed_trials['value'].min()),
            'max_accuracy': float(completed_trials['value'].max())
        }
    }
    
    # Save summary to JSON in training_results directory
    summary_file = os.path.join(save_dir, 'optimization_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Create text report in training_results directory
    report_file = os.path.join(save_dir, 'optimization_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Handwritten Character Recognition - Optuna Optimization Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Optimization Summary:\n")
        f.write(f"  Total trials: {summary['optimization_summary']['total_trials']}\n")
        f.write(f"  Completed trials: {summary['optimization_summary']['completed_trials']}\n")
        f.write(f"  Best trial number: {summary['optimization_summary']['best_trial_number']}\n")
        f.write(f"  Best validation accuracy: {summary['optimization_summary']['best_validation_accuracy']:.4f}\n")
        f.write(f"  Final training accuracy: {summary['optimization_summary']['final_training_accuracy']:.4f}\n")
        f.write(f"  Accuracy improvement: {summary['optimization_summary']['improvement']:.4f}\n\n")
        
        f.write("Best Parameters:\n")
        for key, value in summary['best_parameters'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Trial Statistics:\n")
        f.write(f"  Mean accuracy: {summary['statistics']['mean_accuracy']:.4f}\n")
        f.write(f"  Accuracy standard deviation: {summary['statistics']['std_accuracy']:.4f}\n")
        f.write(f"  Minimum accuracy: {summary['statistics']['min_accuracy']:.4f}\n")
        f.write(f"  Maximum accuracy: {summary['statistics']['max_accuracy']:.4f}\n")
    
    print(f"Optimization summary saved:")
    print(f"  JSON format: {summary_file}")
    print(f"  Text report: {report_file}")
    
    # Display summary
    print(f"\nOptimization Summary:")
    print(f"   Total trials: {summary['optimization_summary']['total_trials']}")
    print(f"   Best validation accuracy: {summary['optimization_summary']['best_validation_accuracy']:.4f}")
    print(f"   Final training accuracy: {summary['optimization_summary']['final_training_accuracy']:.4f}")
    print(f"   Accuracy improvement: {summary['optimization_summary']['improvement']:.4f}")

def objective(trial):
    """Optuna objective function with mixed precision training"""
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128] if use_gpu else [8, 16, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    scheduler_type = trial.suggest_categorical('scheduler', ['StepLR', 'ExponentialLR', 'CosineAnnealingLR'])
    
    if scheduler_type == 'StepLR':
        step_size = trial.suggest_int('step_size', 2, 8)
        gamma = trial.suggest_float('gamma', 0.1, 0.9)
    else:
        step_size = 2
        gamma = trial.suggest_float('gamma', 0.5, 0.95)
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Initialize gradient scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Reload data (using new batch_size)
    train_loader, val_loader, train_size, val_size, class_num, class_names = loaddata(data_dir, batch_size, subsample_rate=subsample_rate)
    
    # Build model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    
    # Add Dropout
    model._fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, class_num)
    )
    
    model = model.to(device)
    
    # Set optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # Set learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, scheduler_type, step_size, gamma)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Simplified training (reduced epochs to 3)
    num_epochs_optuna = 3
    best_acc = train_model_optuna(model, criterion, optimizer, scheduler, 
                                 train_loader, val_loader, train_size, val_size, 
                                 num_epochs_optuna)
    
    return best_acc

if __name__ == "__main__":
    main()