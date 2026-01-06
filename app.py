import os
from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static/results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_PATH = "/Users/usmanhussain/runs/detect/cashew_multi_pest7/weights/best.pt"
model = YOLO(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # Run YOLO detection
            model.predict(
                source=upload_path,
                imgsz=640,
                conf=0.25,
                save=True,
                project=RESULT_FOLDER,
                name="predict",
                exist_ok=True
            )

            # Get the latest file in the predict folder
            predict_folder = os.path.join(RESULT_FOLDER, "predict")
            detected_files = sorted(
                [f for f in os.listdir(predict_folder) if f.endswith((".jpg", ".png"))],
                key=lambda x: os.path.getmtime(os.path.join(predict_folder, x)),
                reverse=True
            )

            if detected_files:
                image_path = os.path.join("results", "predict", detected_files[0])
                print("Detected image path:", image_path)

    return render_template("index.html", image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
