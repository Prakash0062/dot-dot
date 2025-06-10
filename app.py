import io
import base64
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
try:
    # YOLO मॉडल को CPU पर लोड करें
    model = YOLO("best(2).pt")
    logging.info("YOLO model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {str(e)}")
    raise e

def detect_braille(img_bytes, conf_threshold=0.25):
    try:
        logging.debug("Starting braille detection")
        
        # इमेज को रीसाइज़ करें ताकि मेमोरी यूज़ कम हो
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((640, 640))  # 640x640 रीसाइज़
        img_np = np.array(img)
        logging.debug("Image processed, running model prediction")

        # YOLO मॉडल के साथ डिटेक्शन
        results = model(img_np, conf=conf_threshold)
        logging.debug("Model prediction completed")

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = round(float(box.conf[0]), 2)
                label = model.names[int(box.cls[0])]
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'label': label,
                    'x_center': x_center,
                    'y_center': y_center
                })

        if not detections:
            logging.info("No braille detected")
            return "", []

        # डिटेक्शन्स को Y-कोऑर्डिनेट के आधार पर सॉर्ट करें
        detections.sort(key=lambda d: d['y_center'])

        # रो थ्रेशोल्ड के आधार पर डिटेक्शन्स को ग्रुप करें
        row_thresh = 20
        rows, current_row = [], []
        last_y = -100

        for det in detections:
            y = det['y_center']
            if abs(y - last_y) > row_thresh:
                if current_row:
                    rows.append(current_row)
                current_row = [det]
                last_y = y
            else:
                current_row.append(det)
                last_y = (last_y + y) // 2

        if current_row:
            rows.append(current_row)

        # प्रत्येक रो से टेक्स्ट निकालें
        detected_text_rows = []
        for row in rows:
            sorted_row = sorted(row, key=lambda d: d['x_center'])
            row_labels = [d['label'] for d in sorted_row]
            detected_text_rows.append(''.join(row_labels))

        # डिटेक्शन बॉक्स और लेबल्स ड्रॉ करें
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            label = det['label']
            color = (0, 255, 0)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, f'{label} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # इमेज को बेस64 में कन्वर्ट करें
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')
        logging.debug("Braille detection completed successfully")

        return img_str, detected_text_rows
    except Exception as e:
        logging.error(f"Error in detect_braille: {str(e)}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            logging.warning("No image uploaded")
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        conf_threshold = float(request.form.get('confidence', 0.25))

        # इमेज साइज़ चेक करें (5MB लिमिट)
        if len(image_bytes) > 5 * 1024 * 1024:
            logging.warning("Image size exceeds 5MB")
            return jsonify({'error': 'Image size exceeds 5MB'}), 400

        detected_image, detected_text_rows = detect_braille(image_bytes, conf_threshold)
        if not detected_image:
            logging.info("No Braille detected")
            return jsonify({'error': 'No Braille detected'}), 200

        return jsonify({
            'detected_image': f'data:image/jpeg;base64,{detected_image}',
            'detected_text_rows': detected_text_rows
        })
    except Exception as e:
        logging.error(f"Error in /detect endpoint: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
