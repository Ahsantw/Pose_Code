# Video Inference

### Requirement
* Python 3.8
* Cython
* Pytorch 1.11
* Torchvision 0.12.0
* Numpy
* ultralytics

### Pose Model
* [Model](https://drive.google.com/file/d/1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn/view)

### Person Detector Model
* [Model](https://github.com/Ahsantw/Pose_Code/blob/main/Inference/Custom_Detector_Yolov8n.pt)

### Command
```
python video_inference.py --cfg 256x192_res50_lr1e-3_1x.yaml --checkpoint fast_res50_256x192.pth --video examples_media_video.avi --yolo_model Custom_Detector_Yolov8n.pt --save_video --output_video_name output.mp4
```
* --video: Input Video Path
* --checkpoint: Pose Model Path
* --output_video_name: Output Video Path
* --yolo_model: Detector Model Path
