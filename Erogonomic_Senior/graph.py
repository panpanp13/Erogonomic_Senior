import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (use the appropriate model file path)
model = YOLO('yolov8n.pt')  # Make sure to specify the correct path to your model

# Initialize the webcam or video source
cap = cv2.VideoCapture(1)  # 0 for webcam, replace with path if using a video file

# Check if the video source is opened correctly
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Each result contains boxes (bounding boxes, confidences, and class IDs)
    for result in results:
        boxes = result.boxes  # Boxes containing the bounding boxes and other info
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()  # Extract bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID

            # Check if the detected object is a car (class_id = 2 for COCO)
            if class_id == 2:
                label = f"Car {conf:.2f}"
                color = (0, 255, 0)  # Green color for car

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

                # Add the label text on the frame
                cv2.putText(frame, label, (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the frame with bounding boxes
    cv2.imshow('YOLO Car Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close windows
cap.release()
cv2.destroyAllWindows()
