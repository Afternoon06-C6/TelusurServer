from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import os
from ultralytics import YOLO
import tempfile
import uuid
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import threading

app = Flask(__name__)
CORS(app)

# Configuration
APP_NAME = "Telusur"
PORT = 4789

BASE_DIR = Path.home() / "Library" / "Application Support" / APP_NAME
UPLOAD_FOLDER = BASE_DIR / "Uploads"
PROCESSED_FOLDER = BASE_DIR / "Processed"

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
CHOSEN_COLOR = "blue"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Analysis interval for frame processing: set to 1 to analyze every frame,
# or increase to 5, 10, etc. to analyze every Nth frame for faster processing.
ANALYSIS_INTERVAL = 15

# Load YOLO model
try:
    model = YOLO("yolo11s.pt")  # Download automatically if not present
    tshirt_model = YOLO("tshirt_detection_model.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None
    tshirt_model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_with_yolo(input_path, output_path):
    """Process video with YOLO detection and save with bounding boxes using two-stage detection pipeline"""
    if model is None or tshirt_model is None:
        raise Exception("YOLO or t-shirt model not loaded")

    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0  # Ensure frame_count is set before loop
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_annotated_frame = None
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame = frame.copy()

        if frame_count % ANALYSIS_INTERVAL == 0:
            # Clear last detections
            last_detections.clear()

            # Stage 1: Detect persons in the frame
            results = model(frame)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Filter for 'person' class (class 0 for COCO YOLO)
                    if hasattr(box, "cls"):
                        class_id = (
                            int(box.cls[0])
                            if hasattr(box.cls, "__getitem__")
                            else int(box.cls)
                        )
                    else:
                        class_id = int(box[5]) if len(box) > 5 else 0

                    if class_id != 0:
                        continue

                    # Get bounding box coordinates
                    if hasattr(box, "xyxy"):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    else:
                        x1, y1, x2, y2 = [int(b) for b in box[:4]]

                    # Crop ROI for the detected person
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    # Stage 2: Detect t-shirt in the ROI
                    tshirt_results = tshirt_model(roi)
                    tshirt_boxes = tshirt_results[0].boxes

                    if tshirt_boxes is not None and len(tshirt_boxes) > 0:
                        for tshirt_box in tshirt_boxes:
                            # Get tshirt box coordinates relative to ROI
                            if hasattr(tshirt_box, "xyxy"):
                                tx1, ty1, tx2, ty2 = tshirt_box.xyxy[0].cpu().numpy().astype(int)
                            else:
                                tx1, ty1, tx2, ty2 = [int(b) for b in tshirt_box[:4]]

                            # Convert to absolute coordinates relative to main frame
                            abs_x1 = x1 + tx1
                            abs_y1 = y1 + ty1
                            abs_x2 = x1 + tx2
                            abs_y2 = y1 + ty2

                            # Get confidence and label
                            conf = float(tshirt_box.conf[0]) if hasattr(tshirt_box, "conf") else 0.0
                            label = tshirt_model.names[int(tshirt_box.cls[0])] if hasattr(tshirt_box, "cls") else "tshirt"

                            if label.lower() == CHOSEN_COLOR.lower():
                                last_detections.append((abs_x1, abs_y1, abs_x2, abs_y2, label, conf))

            last_annotated_frame = annotated_frame.copy()
        else:
            if last_annotated_frame is not None:
                annotated_frame = frame.copy()

        # Draw all detections on annotated_frame
        for (x1, y1, x2, y2, label, conf) in last_detections:
            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Write frame
        out.write(annotated_frame)

        print(f"Processing frame {frame_count}/{total_frames}")

    cap.release()
    out.release()

    return True


@app.route("/upload", methods=["POST"])
def upload_videos():
    """Upload and process multiple videos"""
    if "videos" not in request.files:
        return jsonify({"error": "No videos provided"}), 400

    files = request.files.getlist("videos")
    threads = []
    results = []

    def process_file(file, unique_id, original_filename, input_path, output_path):
        try:
            # Save uploaded file
            file.save(input_path)

            # Process with YOLO
            process_video_with_yolo(input_path, output_path)

            results.append(
                {
                    "original_name": original_filename,
                    "processed_id": unique_id,
                    "processed_filename": os.path.basename(output_path),
                }
            )

            # Clean up input file
            os.remove(input_path)

        except Exception as e:
            results.append({"error": f"Error processing {file.filename}: {str(e)}"})

    for file in files:
        if file and file.filename and allowed_file(file.filename):
            unique_id = str(uuid.uuid4())
            original_filename = secure_filename(file.filename)
            input_filename = f"{unique_id}_{original_filename}"
            output_filename = f"processed_{unique_id}_{original_filename}"

            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)

            t = threading.Thread(target=process_file, args=(file, unique_id, original_filename, input_path, output_path))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    return jsonify(
        {"message": "Videos processed successfully", "processed_files": results}
    )


@app.route("/video/<video_id>")
def get_processed_video(video_id):
    """Serve processed video file"""
    try:
        # Find the video file with the given ID
        for filename in os.listdir(PROCESSED_FOLDER):
            if video_id in filename:
                video_path = os.path.join(PROCESSED_FOLDER, filename)
                return send_file(video_path, mimetype="video/mp4")

        return jsonify({"error": "Video not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({"status": "healthy", "yolo_model": model_status})


if __name__ == "__main__":
    print("Starting Flask server...")
    print(f"YOLO model status: {'Loaded' if model else 'Failed to load'}")
    app.run(debug=True, host="0.0.0.0", port=PORT)
