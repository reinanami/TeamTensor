import os
import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 
import matplotlib.pyplot as plt
from PIL import Image

train_image_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/train/images'
train_label_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/train/labels'

test_image_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/test/images'
test_label_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/test/labels'

'''
**************************************** PLEASE READ ****************************************************
HOW TO HAVE ALL DEPENDENCIES INSTALLED:
1. Go to your commandline
2. Copy and paste this line: pip install torch torchvision opencv-python numpy matplotlib

HOW TO GET YOUR KAGGLE INSTALLED:
1. If you want to download this Kaggle training data: https://www.kaggle.com/c/111-1-ntut-dl-app-hw4/code
2. Put it in dataset/human_detection_in_drone_videos
3. Download this Kaggle training data as well: https://www.kaggle.com/datasets/rgbnihal/c2a-dataset
4. Put it in dataset/human_deection_disaster_scenarios
5. Modify the code as needed!

# Documentation Links:
# - PyTorch: https://pytorch.org/docs/stable/index.html
# - OpenCV: https://docs.opencv.org/master/
# - Matplotlib: https://matplotlib.org/stable/contents.html

*********************************************************************************************************
'''

'''
************************* SIZE, PARAMETERS, LOSS FUNCTION, OPTIMIZATION *********************************
'''
# Load dataset function
def load_dataset(image_folder, label_folder):
    # TODO: Implement dataset loading and preprocessing
    pass

# Model configuration
IMG_SIZE = (416, 416)  # Standard YOLO input size
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

# Loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam

def plot_training_loss(losses):
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

# Define YOLOv3 Model
def build_yolov3():
    # TODO: Define or load YOLOv3 architecture
    model = None  # Placeholder
    return model

'''
*********************************************************************************************************
'''

'''
************************************ COMPILE AND TRAINING THE MODEL *************************************
'''
# Compile model
model = build_yolov3()
if model:
    optimizer = optimizer(model.parameters(), lr=LEARNING_RATE)

def train_model():
    print("Starting training...")
    training_losses = []
    for epoch in range(EPOCHS):
        # TODO: Implement actual training logic
        loss = np.random.random()  # Placeholder for actual loss
        training_losses.append(loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")
    print("Training complete!")
    plot_training_loss(training_losses)

def evaluate_model():
    print("Evaluating model...")
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

# Run training
if __name__ == "__main__":
    train_model()
    evaluate_model()
    run_camera()

'''
*********************************************************************************************************
'''
