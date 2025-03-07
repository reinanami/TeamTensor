import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

# Dataset Paths
train_image_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/train/images'
train_label_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/train/labels'

test_image_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/test/images'
test_label_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/test/labels'

'''
************************ DEBUG/PATCH NOTES AS OF 03/07/2025 PLEASE READ *********************************

# BROKEN YOLOV3: I'm not sure how to use YOLO V3 and I'm still learning so if someone could lend their
# hand that would be so helpful

# Broken Loss: I think this is related to the broken YOLOV3. Please try to replace the placeholder with
# the actual YOLOV3 and send me the proper documentations because I am lost

# Please get the tasks above done by 5 pm tonight or otherwise we are cooked.

# ChatGPT is useless and unhelpful. Please rely on the documentations as much as we can!


*********************************************************************************************************
'''
'''
************************* SIZE, PARAMETERS, LOSS FUNCTION, OPTIMIZATION *********************************
'''
# Model Parameters
IMG_SIZE = (416, 416)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

torch.manual_seed(42)  # Ensuring consistent dataset generation

# HELP: Please modify the load dataset, I'm not sure how YOLO really works
def load_dataset():
    dataset_size = 100  # <- PLACEHOLDER
    x_data = torch.rand((dataset_size, 3, *IMG_SIZE))  # Random Imges from the dataset
    y_data = torch.randint(0, 2, (dataset_size, 3, *IMG_SIZE), dtype=torch.float32)  # Random binary targets
    x_data, y_data = x_data / 255.0, y_data  # Normalize input images
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# HELP: Replace with YOLO because YOLO wasn't working I put a placeholder
def build_yolov3():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # Output layer
        nn.Sigmoid()  # Ensure output is between 0 and 1
    )
    return model

# Initialize Model
model = build_yolov3() # HELP: Replace later
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss()

'''
*********************************************************************************************************
'''
'''
************************************ COMPILE AND TRAINING THE MODEL *************************************
'''

def plot_training_loss(losses):
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

def train_model():
    print("Starting training...")
    model.train()  # Set model to training mode
    training_losses = []
    dataloader = load_dataset()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    plot_training_loss(training_losses)
    torch.save(model.state_dict(), "training.pth")  # Save model
    print("Model saved as 'training.pth'")

def evaluate_model():
    print("Evaluating model...")
    model.eval()  # Set model to evaluation mode
    # TODO: Implement evaluation process
    print("Evaluation complete!")

def run_camera():
    print("Starting camera for real-time detection...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # TODO: Add YOLO model inference here
        cv2.imshow("YOLOv3 Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

'''
*********************************************************************************************************
'''
'''
************************************************ MAIN ***************************************************
'''
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Train the model") # Do while training needs to be done
    print("2. Use the camera for real-time detection") # Do after training, do not mess it up.
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        train_model()
        evaluate_model()
    elif choice == '2':
        run_camera()
    else:
        print("Invalid input. Please enter 1 or 2.")
'''
*********************************************************************************************************
'''
