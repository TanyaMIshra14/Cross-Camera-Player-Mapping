
Cross-Camera Player Mapping Using YOLO and Feature Matching

This project performs **cross-camera player mapping** using object detection and feature similarity across frames from two synchronized video feeds. It works entirely within Google Colab.

Notebook

- `CrossMapping.ipynb`: A Colab-compatible Jupyter notebook containing all logic for:
  - Player detection using YOLOv8
  - Feature extraction using ResNet
  - Cross-camera player ID assignment based on similarity
  - Visualization of detections and matches

Setup Instructions (Google Colab Only)

1. Upload the Notebook  
   Upload `CrossMapping.ipynb` to your Colab environment.

2. Install Required Dependencies

   Run this cell in Colab before executing the rest of the notebook:

   !pip install ultralytics
   
3. Run the Notebook Cells in Order 
   Follow the order of cells from top to bottom. The notebook will guide you through:
   - Uploading videos
   - Processing a limited number of frames (default: 300)
   - Generating side-by-side visualizations
   - Saving results to `results.json` and `output.mp4`

Dependencies

Install the following inside Colab (most are pre-installed):

!pip install ultralytics

The notebook uses the following libraries:
- `cv2` (OpenCV)
- `torch`, `torchvision`
- `numpy`, `scipy`
- `matplotlib`
- `IPython.display`
- `os`, `json`, `time`, `warnings`, `dataclasses`

How to Use
After executing all cells:
1. Upload two videos: one from a "tacticam" and one from a "broadcast".
2. It processes both feeds and displays re-identified players.
3. Outputs saved:
   - `output.mp4`: Side-by-side annotated video
   - `results.json`: Detection and matching summary

Optional Debugging
To debug a specific frame:

debug_single_frame("tacticam.mp4", "broadcast.mp4", frame_num=100)

 Output

- `output.mp4`: Visual output with bounding boxes and IDs
- `results.json`: JSON log of matches, IDs, and confidence scores
- `debug_frame_*.jpg`: Frame-level image visualizations (optional debug)

Notes

Ensure uploaded videos are of decent quality and synchronized.
Ideal for sports analytics or multi-camera surveillance.
Performance may vary with camera angle, lighting, and occlusions.
