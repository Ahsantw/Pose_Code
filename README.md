# Human Pose Estimation

The code has three major parts.
* Video Inference for Pose Estimation.
* Data Preparation for Human Detector.
* Training of Human Detector.

### Flowchart
![0](https://github.com/Ahsantw/Pose_Code/blob/main/sample/sample_block_diagram.png)


### Sample Output.
![1](https://github.com/Ahsantw/Pose_Code/blob/main/sample/1.gif)
![2](https://github.com/Ahsantw/Pose_Code/blob/main/sample/2.gif)

### Video Inference
The human detector model will predict human from the input frame and then pass it to pose estimation model. The flow is provided in diagram below.
![4](https://github.com/Ahsantw/Pose_Code/blob/main/sample/sample_process.png)
Please follow the [guide](https://github.com/Ahsantw/Pose_Code/tree/main/Inference) for video inference on your own video.

### Dataset Preparation
The data sets we used were sourced from reputable sources such as COCO, MPII, Kaggle, GitHub, and included CCTV footages from various origins. Please follow the steps mentioned in the [readme](https://github.com/Ahsantw/Pose_Code/blob/main/Data_Preparation/README.md) to prepare your dataset in coco and yolo format.

### Training.

The configuration employing YOLOv8n detection model for transfer learning yielded superior results. Employing a batch size of 16 and training over 10 epochs, with an initial learning rate set at 0.01, resulted in the attainment of the most favorable outcomes. We trained approximately 30K high resolution images and 400 bad quality images acquired randomly and labeled.

Custom Training Graphs.
![graph1](https://github.com/Ahsantw/Pose_Code/blob/main/sample/output.png)

Please use [code](https://github.com/Ahsantw/Pose_Code/tree/main/Training_code) for the training the human detector model.
