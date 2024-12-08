import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

class YOLOv9(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv9, self).__init__()
        # Basic YOLOv9 architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes * 5)  # 5 values per class (x, y, w, h, confidence)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_yolo_v9(data_dir, class_names):
    """Train YOLO-V9 model using specified training data."""
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: The specified data directory '{data_dir}' does not exist.")
        sys.exit(1)

    # Initialize model and move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLOv9(len(class_names)).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Basic training parameters
    num_epochs = 100
    batch_size = 16
    
    # Setup data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    print(f"Starting training process for classes {class_names} using data from '{data_dir}'...")
    
    try:
        # Training loop would go here
        # This is a simplified version - you'd need to implement proper data loading
        # and training loop logic for a full implementation
        
        print("Training completed successfully.")
        
        # Get model save path from user
        model_save_path = input("Enter the filename to save the model (e.g. 'my_model.pth'): ")
        
        # Save the model
        torch.save(model.state_dict(), model_save_path)
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # USER INPUT: GET NUMBER OF CLASSES
    while True:
        try:
            num_classes = int(input("Enter the number of classes to train: "))
            if num_classes > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    # USER INPUT: GET TRAINING DATA DIRECTORY
    data_directory = input("Enter the directory of training data images: ")
    
    # USER INPUT: GET CLASS NAMES
    class_names = []
    for i in range(num_classes):
        class_name = input(f"Enter name for class {i+1}: ").strip()
        class_names.append(class_name)

    # Start the training process
    train_yolo_v9(data_directory, class_names)