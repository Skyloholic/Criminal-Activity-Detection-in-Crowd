from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")  # or "last.pt"

# Predict on an image
detection_output = model.predict(source="inference/images/handgun.jpg", conf=0.25, save=True)

# Display detection results
for result in detection_output:
    # Print the detected classes
    boxes = result.boxes  # get bounding boxes
    for box in boxes:
        class_id = int(box.cls.numpy()[0])
        confidence = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]  # bounding box coordinates
        print(f"Detected class: {class_id} with confidence: {confidence} at {bb}")

# Optionally visualize the results
# You can use cv2 to draw bounding boxes on the original image if desired
