# Soccer Player Re-Identification and Tracking with YOLO + DeepSORT

<p align="center">
  <img src="https://img.icons8.com/color/48/000000/python.png" alt="Python" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/13/OpenCV_Logo_with_text_svg_version.svg" alt="OpenCV" height="48"/>
  <img src="https://user-images.githubusercontent.com/36268245/236681171-43c7bd34-b5e6-468f-b7e5-1a20fb1aa1e0.png" alt="YOLOv8" height="48"/>
  <img src="https://camo.githubusercontent.com/f191f26dce46c5ed2183707f4b1e10436f2b4d52d4c8a6e1c3a8a104fae56a47/68747470733a2f2f64656570736f72742e726561646d652e696f2f696d616765732f64656570736f72745f6c6f676f2e706e67" alt="DeepSORT" height="48"/>
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
