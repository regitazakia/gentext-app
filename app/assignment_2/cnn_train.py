# train.py - Part 2: Training the model from Part 1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import CNN  # Import the CNN class from Part 1
import os

# Add this to ensure output is shown immediately
import sys
sys.stdout.flush()

# Set your save directory
SAVE_DIR = '/Users/regita/Documents/Applied Gen AI/Assignment 2/'
MODEL_PATH = os.path.join(SAVE_DIR, 'cnn_cifar10.pth')

def train_model():
    print("=" * 50)
    print("Starting CIFAR10 Training")
    print("=" * 50)
    sys.stdout.flush()
    
    # Verify save directory exists
    if not os.path.exists(SAVE_DIR):
        print(f"Creating directory: {SAVE_DIR}")
        os.makedirs(SAVE_DIR)
    print(f"Model will be saved to: {MODEL_PATH}")
    sys.stdout.flush()
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR10 dataset
    print("\n[1/5] Loading CIFAR10 dataset...")
    sys.stdout.flush()
    
    try:
        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                         download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                        download=True, transform=transform)
        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    sys.stdout.flush()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Initialize model from Part 1
    print("\n[2/5] Initializing model...")
    sys.stdout.flush()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    try:
        model = CNN().to(device)
        print("✓ Model created successfully")
        print(f"\nModel Architecture:")
        print(model)
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return
    
    sys.stdout.flush()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    print(f"\n[3/5] Starting training for {num_epochs} epochs...")
    print("=" * 50)
    sys.stdout.flush()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        sys.stdout.flush()
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'  Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                sys.stdout.flush()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'  Epoch Summary: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%')
        sys.stdout.flush()
    
    # Save model to your specified directory
    print(f"\n[4/5] Saving model...")
    sys.stdout.flush()
    
    try:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'✓ Model saved to: {MODEL_PATH}')
        
        # Verify file was created
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Size in MB
            print(f'✓ File size: {file_size:.2f} MB')
        else:
            print('✗ Warning: File not found after saving!')
    except Exception as e:
        print(f'✗ Error saving model: {e}')
    
    sys.stdout.flush()
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    sys.stdout.flush()
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f'✓ Test Accuracy: {test_acc:.2f}%')
    print("=" * 50)
    print("Training Complete!")
    print("=" * 50)
    sys.stdout.flush()
    
    return test_acc

if __name__ == '__main__':
    try:
        print("Python script started!")
        sys.stdout.flush()
        train_model()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()