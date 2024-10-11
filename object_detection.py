import torch
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

def detect_objects(frame):

    # Convert the frame to RGB (YOLOv5 uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference with YOLOv5
    results = model(rgb_frame)

    # Extract detection results (bounding boxes, confidence, class ID)
    detected_data = results.xyxy[0].cpu().numpy()

    return detected_data, model.names

def draw_detections(frame, detected_data, class_names, distance_calculator=None):

    for detection in detected_data:
        x1, y1, x2, y2, confidence, class_id = detection

        # Only process objects with a confidence score higher than 0.4
        if confidence > 0.4:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Get the object class name
            class_name = class_names[int(class_id)]

            # Optional: Calculate distance and overlay it on the frame
            label = f"{class_name} {confidence:.2f}"

            if distance_calculator:
                # Calculate object height in pixels
                object_height = y2 - y1
                distance = distance_calculator.calculate_distance(object_height)
                label += f" - {distance:.2f}m"

            # Put the label on the frame
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    return frame

