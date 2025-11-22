from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train on redmites dataset
model.train(data="redmites.yaml", epochs=50, imgsz=640)
