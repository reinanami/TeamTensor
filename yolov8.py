import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # YOLOv8 nano 


#ayush@developer · 5mo ago - Function
def detect_and_plot_bounding_box(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    results = model(img)

    for result in results:  # Loop through the detected objects
        boxes = result.boxes  # Access boxes from result
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
            conf = float(box.conf.cpu().numpy())  # Extract confidence as a scalar
            cls = int(box.cls.cpu().numpy().item())  # Extract class index as a scalar
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{model.names[cls]} {conf:.2f}'  
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

detect_and_plot_bounding_box('c2a-dataset/C2A_Dataset/new_dataset3/test/images/traffic_incident_image0153_3.png')
