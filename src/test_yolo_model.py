from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Load test image
test_image = "test_images/redmite.jpeg"   # change filename accordingly

# Run detection
results = model(test_image)

# Show classes and boxes in terminal
for r in results:
    print("Detected classes:", r.names)
    print("Boxes:", r.boxes)

# Save output image
results[0].save("prediction.jpg")
print("✔ Saved prediction as prediction.jpg")
