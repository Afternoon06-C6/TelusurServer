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

app = Flask(__name__)
CORS(app)

# Configuration
APP_NAME = "Telusur"
BASE_DIR = Path.home() / "Library" / "Application Support" / APP_NAME
UPLOAD_FOLDER = BASE_DIR / 'uploads'
PROCESSED_FOLDER = BASE_DIR / 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
PORT=4789

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLO model
try:
    model = YOLO('yolo11s.pt')  # Download automatically if not present
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_with_yolo(input_path, output_path):
    """Process video with YOLO detection and save with bounding boxes"""
    if model is None:
        raise Exception("YOLO model not loaded")
    
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO inference
        results = model(frame)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Write frame
        out.write(annotated_frame)
        
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return True

@app.route('/upload', methods=['POST'])
def upload_videos():
    """Upload and process multiple videos"""
    if 'videos' not in request.files:
        return jsonify({'error': 'No videos provided'}), 400
    
    files = request.files.getlist('videos')
    processed_files = []
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            try:
                # Generate unique filename
                unique_id = str(uuid.uuid4())
                original_filename = secure_filename(file.filename)
                input_filename = f"{unique_id}_{original_filename}"
                output_filename = f"processed_{unique_id}_{original_filename}"
                
                input_path = os.path.join(UPLOAD_FOLDER, input_filename)
                output_path = os.path.join(PROCESSED_FOLDER, output_filename)
                
                # Save uploaded file
                file.save(input_path)
                
                # Process with YOLO
                process_video_with_yolo(input_path, output_path)
                
                processed_files.append({
                    'original_name': original_filename,
                    'processed_id': unique_id,
                    'processed_filename': output_filename
                })
                
                # Clean up input file
                os.remove(input_path)
                
            except Exception as e:
                return jsonify({'error': f'Error processing {file.filename}: {str(e)}'}), 500
    
    return jsonify({
        'message': 'Videos processed successfully',
        'processed_files': processed_files
    })

@app.route('/video/<video_id>')
def get_processed_video(video_id):
    """Serve processed video file"""
    try:
        # Find the video file with the given ID
        for filename in os.listdir(PROCESSED_FOLDER):
            if video_id in filename:
                video_path = os.path.join(PROCESSED_FOLDER, filename)
                return send_file(video_path, mimetype='video/mp4')
        
        return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'yolo_model': model_status
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"YOLO model status: {'Loaded' if model else 'Failed to load'}")
    app.run(debug=True, host='0.0.0.0', port=PORT)
