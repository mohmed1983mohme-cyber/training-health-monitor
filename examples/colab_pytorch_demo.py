"""
Example: Using Training Health Monitor with PyTorch in Colab
"""

import sys
sys.path.insert(0, '/content/training-health-monitor')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our framework
try:
    from training_monitor.monitor import TrainingHealthMonitor
    from training_monitor.utils import extract_gradients_pytorch
    print("✅ Imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Installing from local path...")


class SimpleNet(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def main():
    print("\n" + "="*80)
    print("🚀 TRAINING HEALTH MONITOR - COLAB DEMO")
    print("="*80 + "\n")
    
    # Create synthetic data
    print("Creating synthetic dataset...")
    X_train = torch.randn(1000, 1, 28, 28)
    y_train = torch.randint(0, 10, (1000,))
    X_val = torch.randn(200, 1, 28, 28)
    y_val = torch.randint(0, 10, (200,))
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}\n")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"✅ Model created on device: {device}\n")
    
    # Initialize monitor
    monitor = TrainingHealthMonitor(
        model=model,
        framework='pytorch',
        verbose=True
    )
    
    print("✅ Training Health Monitor initialized!\n")
    
    # Training loop
    num_epochs = 5
    training_history = []
    
    print("="*80)
    print("STARTING TRAINING WITH HEALTH MONITORING")
    print("="*80 + "\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            train_acc += (predictions == y_batch).sum().item() / len(y_batch)
            num_batches += 1
        
        train_loss /= num_batches
        train_acc /= num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, predictions = torch.max(outputs, 1)
                val_acc += (predictions == y_batch).sum().item() / len(y_batch)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Extract gradients
        gradients = extract_gradients_pytorch(model)
        
        # Check health
        health = monitor.check_health(
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics={'accuracy': train_acc},
            val_metrics={'accuracy': val_acc},
            epoch=epoch,
            gradients=gradients
        )
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'status': health.status
        })
        
        # Print summary
        status_emoji = '✅' if health.status == 'healthy' else '⚠️' if health.status == 'warning' else '🔴'
        print(f"Epoch {epoch+1:2d}/{num_epochs} {status_emoji}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Health: {health.status.upper()}")
        print()
    
    print("="*80)
    print("✅ TRAINING COMPLETED")
    print("="*80)
    
    # Get summary
    summary = monitor.get_summary()
    
    print(f"\n📊 TRAINING HEALTH SUMMARY")
    print(f"{'='*60}")
    print(f" Total Epochs:        {summary['total_epochs']}")
    print(f" ✅ Healthy Epochs:   {summary['healthy']}")
    print(f" ⚠️  Warning Epochs:   {summary['warnings']}")
    print(f" 🔴 Critical Epochs:  {summary['critical']}")
    print(f" 📈 Health Score:     {summary['health_percentage']:.1f}%")
    print(f"{'='*60}\n")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Health Monitor - Results', fontsize=16, fontweight='bold')
    
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    train_accs = [h['train_acc'] for h in training_history]
    val_accs = [h['val_acc'] for h in training_history]
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 's-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, train_accs, 'o-', label='Train Acc', linewidth=2)
    ax.plot(epochs, val_accs, 's-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overfitting indicator
    ax = axes[1, 0]
    loss_ratios = [h['val_loss'] / h['train_loss'] if h['train_loss'] > 0 else 0 for h in training_history]
    colors = ['green' if r < 1.2 else 'orange' for r in loss_ratios]
    ax.bar(epochs, loss_ratios, color=colors, alpha=0.7)
    ax.axhline(y=1.2, color='r', linestyle='--', linewidth=2, label='Overfitting Threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss / Train Loss')
    ax.set_title('Overfitting Indicator')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Health distribution
    ax = axes[1, 1]
    statuses = [h['status'] for h in training_history]
    status_counts = {
        'healthy': statuses.count('healthy'),
        'warning': statuses.count('warning'),
        'critical': statuses.count('critical')
    }
    ax.pie([status_counts['healthy'], status_counts['warning'], status_counts['critical']],
           labels=['Healthy', 'Warning', 'Critical'],
           colors=['green', 'orange', 'red'],
           autopct='%1.1f%%')
    ax.set_title('Training Health Distribution')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
