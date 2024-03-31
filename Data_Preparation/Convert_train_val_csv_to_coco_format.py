import pandas as pd
import os
import glob
import json
import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi

csv_list = ['instances_train2017','instances_val2017']


for csv in csv_list:
    h_coco={}
    h_coco["info"] = {"year" : "2024",
                         "version" : "1.0",
                         "description" : "123",
                         "contributor" : "456",
                         "url" : "",
                         "date_created" : "2020"
                        }
    h_coco["licenses"] = [{"id": 1,
                      "name": "123",
                      "url": "123"
                     }]

    h_coco["categories"] =[{'supercategory': 'Human', 'id': 1, 'name': 'Person'}]
    cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(h_coco["categories"] )}
    print(cat2id)

    annot=pd.read_csv(csv+'.csv')
    print(len(annot.Image_Name.unique()))


    coco_images = []
    annotations=[]
    k=0
    i=0
    image_list=[]
    for j in range(len(annot['Image_Name'])):
        
        class_name=annot['Class'][j]
        print(i,k,j)


        height=int(annot['img_height'][j])
        width=int(annot['img_width'][j])
        #print(height,width)

        xmin=int(annot['xmin'][j])
        ymin=int(annot['ymin'][j])
        xmax=int(annot['xmax'][j])
        ymax=int(annot['ymax'][j])
        w=xmax-xmin
        h=ymax-ymin
        area = w*h

        seg = []
           
        if annot['Image_Name'][j] in image_list:
            pass
        else:
            image_list.append(annot['Image_Name'][j])
            coco_images.append({"date_captured" : "2024",
                    "file_name" : annot['Image_Name'][j], 
                    "id" : i+1,
                    "license" : 1,
                    "url" : "",
                    "height" : int(annot['img_height'][j]),
                    "width" : int(annot['img_width'][j])})
                
            i=i+1
            
        annotations.append({"segmentation" : seg,
                            "area" : float(area),
                            "iscrowd" : 0,
                            "image_id" : i,
                            "bbox" : [xmin,ymin,w,h],
                            "category_id" : cat2id[class_name],
                            "id": k+1})
        k=k+1

    h_coco['images']=coco_images
    h_coco['annotations']=annotations

    with open(csv+'.json', 'w') as json_file:
        json.dump(h_coco, json_file, sort_keys=True, indent=4)