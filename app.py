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
from PIL import ImageFont, ImageDraw, Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
APP_NAME = "Telusur"
PORT = 4789

BASE_DIR = Path.home() / "Library" / "Application Support" / APP_NAME
UPLOAD_FOLDER = BASE_DIR / "Uploads"
PROCESSED_FOLDER = BASE_DIR / "Processed"
IMAGES_FOLDER = BASE_DIR / "Images"

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Analysis interval for frame processing: set to 1 to analyze every frame,
# or increase to 5, 10, etc. to analyze every Nth frame for faster processing.
ANALYSIS_INTERVAL = 15
IOU_THRESHOLD = 0.3
MAX_MISSED = 3

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

def compute_iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA + 1)
    interHeight = max(0, yB - yA + 1)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

    return iou

def save_crop_at_frame(video_path, bbox, frame_num, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        cap.release()
        return False
    cv2.imwrite(output_path, crop)
    cap.release()
    return True

def process_video_with_yolo(input_path, output_path, chosen_color, uuid_str=None, original_filename=None):
    """Process video with YOLO detection and save with bounding boxes using two-stage detection pipeline"""
    if model is None or tshirt_model is None:
        raise Exception("YOLO or t-shirt model not loaded")

    cap = cv2.VideoCapture(input_path)

    image_files = []

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

    active_tracks = {}
    finished_tracks = []
    next_track_id = 0

    # current_detections contains detections to draw for non-analysis frames
    current_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame = frame.copy()

        updated_track_ids = set()

        # By default keep previous detections for drawing when not analyzing
        detections_for_matching = []

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
                    # clamp coordinates to frame to avoid empty rois
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width - 1, x2))
                    y2 = max(0, min(height - 1, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

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

                            # Clamp to frame bounds
                            abs_x1 = max(0, min(width - 1, abs_x1))
                            abs_y1 = max(0, min(height - 1, abs_y1))
                            abs_x2 = max(0, min(width - 1, abs_x2))
                            abs_y2 = max(0, min(height - 1, abs_y2))

                            # Get confidence and label
                            conf = float(tshirt_box.conf[0]) if hasattr(tshirt_box, "conf") else 0.0
                            label = tshirt_model.names[int(tshirt_box.cls[0])] if hasattr(tshirt_box, "cls") else "tshirt"

                            if label.lower() == chosen_color.lower():
                                detections_for_matching.append((abs_x1, abs_y1, abs_x2, abs_y2, label, conf))

            # If we have detections, update matching; otherwise we'll just age tracks
            if len(detections_for_matching) > 0:
                # Match detections to active tracks using IoU
                for det in detections_for_matching:
                    x1, y1, x2, y2, label, conf = det
                    matched_track_id = None
                    max_iou = 0.0
                    for tid, track in list(active_tracks.items()):
                        iou = compute_iou(track['bbox'], (x1, y1, x2, y2))
                        if iou > IOU_THRESHOLD and iou > max_iou:
                            max_iou = iou
                            matched_track_id = tid
                    if matched_track_id is not None:
                        # Update existing track
                        active_tracks[matched_track_id]['bbox'] = (x1, y1, x2, y2)
                        active_tracks[matched_track_id]['last_frame'] = frame_count
                        active_tracks[matched_track_id]['missed'] = 0
                        updated_track_ids.add(matched_track_id)
                    else:
                        # Create new track
                        active_tracks[next_track_id] = {
                            'bbox': (x1, y1, x2, y2),
                            'start_frame': frame_count,
                            'last_frame': frame_count,
                            'label': label,
                            'missed': 0,
                            'saved': False,
                        }
                        updated_track_ids.add(next_track_id)
                        next_track_id += 1

            # Age tracks that were not updated in this analysis frame
            for tid, track in list(active_tracks.items()):
                if tid not in updated_track_ids:
                    track['missed'] = track.get('missed', 0) + 1

            # Determine ended tracks (missed too many analysis cycles)
            ended_track_ids = [tid for tid, track in active_tracks.items() if track.get('missed', 0) > MAX_MISSED]
            for tid in ended_track_ids:
                track = active_tracks[tid]
                # Save images only once per track
                if uuid_str is not None and original_filename is not None and not track.get('saved', False):
                    start_time = track['start_frame'] / fps
                    end_time = track['last_frame'] / fps
                    # Format times for filename (replace '.' with '_')
                    start_time_str = str(round(start_time, 2)).replace('.', '_')
                    end_time_str = str(round(end_time, 2)).replace('.', '_')
                    start_img_name = f"{uuid_str}_{original_filename}_{tid}_{start_time_str}_startFrame.jpg"
                    end_img_name = f"{uuid_str}_{original_filename}_{tid}_{end_time_str}_endFrame.jpg"
                    start_img_path = os.path.join(IMAGES_FOLDER, start_img_name)
                    end_img_path = os.path.join(IMAGES_FOLDER, end_img_name)
                    # Check and create crops only if not existing
                    if not os.path.exists(start_img_path):
                        save_crop_at_frame(input_path, track['bbox'], track['start_frame'], start_img_path)
                    if not os.path.exists(end_img_path):
                        save_crop_at_frame(input_path, track['bbox'], track['last_frame'], end_img_path)
                    track['saved'] = True
                    image_files.append({
                        "start": start_img_name,
                        "end": end_img_name,
                        "start_time": start_time,
                        "end_time": end_time
                    })
                finished_tracks.append(track)
                del active_tracks[tid]

            # store current detections for drawing on non-analysis frames
            current_detections = detections_for_matching.copy()

            last_annotated_frame = annotated_frame.copy()
        else:
            # Not an analysis frame: keep previous annotated frame but still draw last known detections if any
            if last_annotated_frame is not None:
                annotated_frame = frame.copy()

        # Draw all detections on annotated_frame using Pillow for SF Pro text and fixed bounding box colors
        if current_detections:
            # Define color mapping
            color_map = {
                "black": (0, 0, 0),
                "blue": (0, 102, 204),
                "grey": (128, 128, 128),
                "white": (255, 255, 255),
            }
            color = color_map.get(chosen_color.lower(), (255, 0, 0))
            # Convert OpenCV image (BGR) to PIL image (RGB)
            pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("SF-Pro-Text-Regular.otf", 18)
            except Exception:
                font = ImageFont.load_default()
            for (x1, y1, x2, y2, label, conf) in current_detections:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                text = f"{label} {conf:.2f}"
                text_size = draw.textbbox((0, 0), text, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                # Draw filled rectangle for text background with more margin from box
                draw.rectangle([x1, y1 - text_height - 8, x1 + text_width + 4, y1], fill=color)
                # Draw text in white (or black if box color is white)
                text_fill = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)
                draw.text((x1 + 2, y1 - text_height - 6), text, font=font, fill=text_fill)
            # Convert back to OpenCV image (BGR)
            annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Write frame
        out.write(annotated_frame)

        print(f"Processing frame {frame_count}/{total_frames}")

    # Finalize remaining active tracks at end of video
    for tid, track in active_tracks.items():
        if uuid_str is not None and original_filename is not None and not track.get('saved', False):
            start_time = track['start_frame'] / fps
            end_time = track['last_frame'] / fps
            start_time_str = str(round(start_time, 2)).replace('.', '_')
            end_time_str = str(round(end_time, 2)).replace('.', '_')
            start_img_name = f"{uuid_str}_{original_filename}_{tid}_{start_time_str}_startFrame.jpg"
            end_img_name = f"{uuid_str}_{original_filename}_{tid}_{end_time_str}_endFrame.jpg"
            start_img_path = os.path.join(IMAGES_FOLDER, start_img_name)
            end_img_path = os.path.join(IMAGES_FOLDER, end_img_name)
            if not os.path.exists(start_img_path):
                save_crop_at_frame(input_path, track['bbox'], track['start_frame'], start_img_path)
            if not os.path.exists(end_img_path):
                save_crop_at_frame(input_path, track['bbox'], track['last_frame'], end_img_path)
            track['saved'] = True
            image_files.append({
                "start": start_img_name,
                "end": end_img_name,
                "start_time": start_time,
                "end_time": end_time
            })
        finished_tracks.append(track)

    cap.release()
    out.release()

    return image_files


@app.route("/upload", methods=["POST"])
def upload_videos():
    """Upload and process multiple videos"""
    if "videos" not in request.files:
        return jsonify({"error": "No videos provided"}), 400

    # Read uuid and topColor from request.form
    request_uuid = request.form.get("uuid")
    top_color = request.form.get("topColor")

    if not request_uuid:
        return jsonify({"error": "UUID not provided"}), 400
    if not top_color:
        return jsonify({"error": "topColor not provided"}), 400

    files = request.files.getlist("videos")
    threads = []
    results = []

    def process_file(file, uuid_str, original_filename, input_path, output_path, chosen_color):
        try:
            # Save uploaded file
            file.save(input_path)

            # Process with YOLO
            image_files = process_video_with_yolo(input_path, output_path, chosen_color, uuid_str=uuid_str, original_filename=original_filename)

            results.append(
                {
                    "original_name": original_filename,
                    "processed_id": uuid_str,
                    "processed_filename": os.path.basename(output_path),
                    "images": image_files,
                }
            )

            # Clean up input file
            os.remove(input_path)

        except Exception as e:
            results.append({"error": f"Error processing {file.filename}: {str(e)}"})

    for file in files:
        if file and file.filename and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            input_filename = f"{request_uuid}_{original_filename}"
            output_filename = f"processed_{request_uuid}_{original_filename}"

            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)

            t = threading.Thread(target=process_file, args=(file, request_uuid, original_filename, input_path, output_path, top_color))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    return jsonify(
        {"message": "Videos processed successfully", "processed_files": results}
    )

@app.route("/health")
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({"status": "healthy", "yolo_model": model_status})

if __name__ == "__main__":
    print("Starting Flask server...")
    print(f"YOLO model status: {'Loaded' if model else 'Failed to load'}")
    app.run(debug=True, host="0.0.0.0", port=PORT)
