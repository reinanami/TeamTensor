import os
import torch
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from ultralytics import YOLO  # ADD: Import YOLO from ultralytics
#from yolov8 import detect_and_plot_bounding_box  # ADD: Import function from yolov8.py

torch.manual_seed(42)  # Ensuring consistent dataset generation

# Dataset Paths
train_image_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/train/images'
train_label_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/train/labels'

test_image_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/test/images'
test_label_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/test/labels'

'''
************************ DEBUG/PATCH NOTES AS OF 03/07/2025 PLEASE READ *********************************

# DELETED: Custom optimizer and loss function (YOLOv8 handles this internally)

# ChatGPT is useless and unhelpful. Please rely on the documentations as much as we can!

# Issue: RuntimeError: Dataset 'c2a-dataset/C2A_Dataset/new_dataset3/train/images' error ❌ 'c2a-dataset/C2A_Dataset/new_dataset3/train/images' does not exist

*********************************************************************************************************
'''

'''
************************* SIZE, PARAMETERS, LOSS FUNCTION, OPTIMIZATION *********************************
'''
# Model Parameters
IMG_SIZE = 640  # YOLOv8 default image size
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001 # Adjust as needed

model = YOLO("yolov8n.pt")  # Load YOLOv8 nano model

'''
************************************ COMPILE AND TRAINING THE MODEL *************************************
'''

def train_model():
    print("Starting YOLOv8 training...")
    model.train(data= train_image_folder, epochs=EPOCHS)  # ADD: Train using YOLOv8 built-in function
    print("Training complete!")
    model.save("yolov8_trained.pt")  # Save trained model


def evaluate_model():
    print("Evaluating YOLOv8 model...")
    metrics = model.val()
    print("Evaluation complete! Results:", metrics)

'''
*********************************************************************************************************
'''

'''
********************************************** RUN CAMERA ************************************************
'''

def run_camera():
    print("Starting real-time detection...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform YOLOv8 inference
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())
                cls = int(box.cls.cpu().numpy().item())
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow("YOLOv8 Real-Time Detection", frame)
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
    print("1. Train the model")
    print("2. Use the camera for real-time detection")
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