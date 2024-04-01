# Data Prep

### Requirement
* Pandas
* cv2
* numpy
* pycocotools

### Input Labelled Data Format
* All images in one folder.
* Annotation csv [format](https://github.com/Ahsantw/Pose_Code/blob/main/Data_Preparation/Bad_Quality_Data_Frames_output_cleaned_height_width.csv).

### Data Processing
* Split the input csv file into train and val set.
```
python Train_Val_Split.py --csv_name Bad_Quality_Data_Frames_output_cleaned_height_width.csv

--csv_name: Path of the annotated csv
```
* Convert the data into coco json file.
```
python Convert_train_val_csv_to_coco_format.py
```
* Convert the coco json file to yolo label file for each image.
```
python coco_to_yolo_format.py
```
* Copy the images.
```
python copy_images.py --image_dir Bad_Quality_Data_Frames

--image_dir: path to image folder
```

Final Dataset path will be ./new_dir/
