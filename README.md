# Soccer Player Re-Identification and Tracking with YOLO + DeepSORT

<p align="center">
  <img src="https://img.icons8.com/color/48/000000/python.png" alt="Python" />
  <img src="https://raw.githubusercontent.com/opencv/opencv/master/doc/opencv-logo.png" alt="OpenCV" height="48"/>
  <img src="https://raw.githubusercontent.com/roboflow-ai/notebooks/main/notebooks/images/track.gif" alt="DeepSORT" height="48"/>
</p>


## About the Project
This project provides a real-time system for soccer player re-identification and tracking.  
It integrates YOLOv8 for high-speed object detection with DeepSORT for tracking players across frames, maintaining consistent IDs.  
It also includes speed estimation and visual trail overlays to analyze player movements.

This system is built to handle:
- Fast camera movements and panning
- Occlusions where players overlap or are temporarily hidden
- Players leaving and re-entering the frame while keeping the same ID

Typical applications include:
- Sports analytics for coaches and analysts
- Broadcast enhancements such as live overlays
- Training insights to improve player performance

## Features
- YOLOv8 model trained specifically for soccer player detection
- DeepSORT multi-object tracking to maintain unique player IDs
- Bounding boxes dynamically sized to detected players
- Real-time speed overlay for each player
- Motion trails to visualize recent movement history
- FPS monitoring to evaluate performance

## Project Structure
re-idproject/
├── models/
│ └── yolo_soccer.pt # YOLO model weights (download separately)
├── videos/
│ └── input.mp4 # Example input video
├── detect_video.py # Main detection and tracking script
├── track_video.py # Additional experiments
├── requirements.txt
└── README.md

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/AdityaPatil2712/soccer-player-tracking.git
    cd soccer-player-tracking
    ```

2. Create and activate a virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate    # On Windows system
    ```

3. Install the required Python packages:
    ```
    pip install -r requirements.txt
    ```

4. Download the YOLOv8 model weights file from:
    ```
    https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
    ```
    and place it inside the `models/` folder as:
    ```
    models/yolo_soccer.pt
    ```

## Usage
Run the main tracking program with:
Adjust the paths in the script if your video or model locations differ.

## Example Output
The system processes the video, assigning each player a unique ID that remains stable across frames.  
It overlays bounding boxes, ID labels, speed measurements, and motion trails to show recent paths.

## Future Work
- Integration of player heatmaps and field coverage analytics
- Detection of passes, shots, or fouls using additional machine learning models
- Live dashboard visualizations using Streamlit or a web framework
- API service to deliver processed analytics data to external systems

## Acknowledgements
- Ultralytics YOLO for advanced object detection
- DeepSORT for multi-object tracking
- OpenCV for video processing and visualization

## License
This project is provided under the MIT License.  
It may be freely modified and distributed with proper attribution.



