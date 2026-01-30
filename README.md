# TelusurServer

**Backend server for Telusur - A CCTV Footage Analysis Application for Lost & Found**

TelusurServer is a Flask-based Python backend that powers the Telusur macOS application (built with SwiftUI). It uses advanced computer vision and YOLO object detection to analyze CCTV footage and identify people wearing specific colored clothing, making it invaluable for lost & found scenarios.

## 🎯 Overview

This server processes video footage to detect and track individuals based on their top clothing color. It employs a two-stage detection pipeline:

1. **Person Detection**: Uses YOLOv11 to detect people in video frames
2. **T-shirt Color Classification**: Uses a custom trained YOLO model to identify clothing colors (black, blue, grey, white)

The system tracks detected individuals across frames, extracts key frames showing when they appear and disappear, and returns processed videos with bounding boxes plus cropped images for easy identification.

## ✨ Features

- **Asynchronous Video Processing**: Upload multiple videos and track processing status via job IDs
- **Color-based Person Tracking**: Detect individuals wearing specific colored tops (black, blue, grey, white)
- **Smart Frame Analysis**: Configurable frame sampling to balance speed vs accuracy
- **Track Persistence**: IoU-based tracking maintains identity across frames
- **Automatic Cropping**: Saves start/end frame crops of detected individuals
- **REST API**: Easy integration with frontend applications
- **Health Monitoring**: Built-in health check endpoint

## 🛠️ Technology Stack

- **Framework**: Flask with CORS support
- **Computer Vision**: OpenCV, Ultralytics YOLO
- **Deep Learning**: YOLOv11 for person detection, custom YOLO model for clothing classification by Abdul Rehman: [Link](https://github.com/AbdulRehman-git/real-time-tshirt-color-detection-yolov8)
- **Image Processing**: Pillow (PIL) for text rendering
- **Concurrency**: Threading for async job processing

## 📋 Prerequisites

- Python 3.8+
- [mise](https://mise.jdx.dev/) (optional but recommended for environment management)
- CUDA-capable GPU (optional, for faster processing)

## 🚀 Setup Instructions

### 1. Install mise (Optional)

If you want to use mise for environment management:

```bash
# Follow instructions at https://mise.jdx.dev/install
mise install
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:

- Flask and flask-cors
- OpenCV (opencv-python)
- Ultralytics YOLO
- PyTorch and torchvision
- Pillow, numpy, and other dependencies

### 3. Required Model Files

Ensure these model files are in the project root:

- `yolo11s.pt` - YOLOv11 model for person detection
- `tshirt_detection_model.pt` - Custom trained model for t-shirt color classification

## 🎬 Running the Server

```bash
python app.py
```

The server will start on port **4789** and models will be loaded lazily on the first request.

## 🧪 Testing

### Run Basic Server Tests

```bash
python test_server.py
```

This tests the health endpoint and basic API functionality.

### Run Full Video Processing Test

```bash
python test_video.py
```

This simulates a complete workflow: uploading a video, tracking job status, and retrieving results.

## 📡 API Endpoints

### Health Check

```
GET /health
```

Returns server status and model loading state.

### Submit Video Processing Job

```
POST /upload
```

**Form Data:**

- `videos`: Video file(s) (mp4, avi, mov, mkv)
- `uuid`: Unique identifier for this analysis session
- `topColor`: Target clothing color (black, blue, grey, white)

**Response:**

```json
{
  "job_id": "uuid-string",
  "status": "submitted",
  "total_videos": 1
}
```

### Check Job Status

```
GET /job/<job_id>/status
```

**Response:**

```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "progress": 0.5,
  "total_videos": 1,
  "processed_videos": 0
}
```

### Get Job Results

```
GET /job/<job_id>/results
```

Returns processed video information and cropped images once job is completed.

## ⚙️ Configuration

Key parameters in `app.py`:

- `ANALYSIS_INTERVAL = 15`: Process every Nth frame (higher = faster but less accurate)
- `IOU_THRESHOLD = 0.3`: Minimum IoU for track matching
- `MAX_MISSED = 3`: Maximum missed detections before ending a track
- `PORT = 4789`: Server port

Output directories (created automatically in `~/Library/Application Support/Telusur/`):

- `Uploads/`: Temporary storage for uploaded videos
- `Processed/`: Processed videos with bounding boxes
- `Images/`: Cropped images of detected individuals

## 🔧 Development

### Project Structure

```
TelusurServer/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── mise.toml                       # mise configuration
├── yolo11s.pt                      # Person detection model
├── tshirt_detection_model.pt       # Clothing color classification model
├── test_server.py                  # Basic server tests
├── test_video.py                   # Full video processing test
└── video1.mp4                      # Sample test video
```

### Key Components

- **Job Management**: Async processing with status tracking using dataclasses and threading
- **Two-Stage Detection**: Person detection → ROI extraction → T-shirt color classification
- **Track Management**: IoU-based tracking with missed detection handling
- **Frame Cropping**: Automatic extraction of start/end frames for each detected track

## 🤝 Integration with Telusur macOS App

This server is designed to work seamlessly with the Telusur SwiftUI macOS application. The macOS app:

1. Uploads CCTV footage via the `/upload` endpoint
2. Polls `/job/<job_id>/status` for progress updates
3. Retrieves processed results from `/job/<job_id>/results`
4. Displays annotated videos and cropped images to help locate missing persons

## 📝 Notes

- Models are loaded lazily on first request to reduce startup time
- Uploaded videos are automatically cleaned up after processing
- Job history is stored in memory (consider Redis/database for production)
- For optimal performance, run on a machine with CUDA-capable GPU

## 📄 License

This project is part of the Telusur lost & found system.

---

**For more information on using mise, see the [mise documentation](https://mise.jdx.dev/docs/).**
