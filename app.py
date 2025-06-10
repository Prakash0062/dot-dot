import io
import os
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, render_template, send_file, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO(r"best(2).pt")

def detect_braille(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(image)
    results = model(img_np)[0]

    detected_labels = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])  # Class ID
        label = model.names[cls_id]  # Class name (like 'A', 'B', etc.)
        detected_labels.append(label)

        # Draw bounding box and label
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert final image
    _, img_encoded = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return io.BytesIO(img_encoded.tobytes()), "".join(detected_labels)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No image file uploaded", 400

    image_file = request.files["image"]
    img_bytes = image_file.read()

    processed_img_io, detected_text = detect_braille(img_bytes)

    # Save image temporarily
    with open("static/output.png", "wb") as f:
        f.write(processed_img_io.getbuffer())

    return jsonify({
        "image_url": "/static/output.png",
        "detected_text": detected_text
    })

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
