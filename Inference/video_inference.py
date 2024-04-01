from ultralytics import YOLO
import cv2

import argparse
import os
import platform
import sys
import time

import numpy as np
import torch

from alphapose.utils.config import update_config
from alphapose.models import builder
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.utils.writer import DataWriter
from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json


"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="./")
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=True, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=2,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=2,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="-1",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=True, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
parser.add_argument('--yolo_model', dest='yolo_model',
                    help='', required=True , default=False)
parser.add_argument('--output_video_name', dest='output_video_name',
                    help='', default='output.mp4')

args = parser.parse_args()
cfg = update_config(args.cfg)

model = YOLO(args.yolo_model)

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow

# Load pose model
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

print('Loading pose model from %s...' % (args.checkpoint,))
pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
pose_model.to(args.device)
pose_model.eval()

runtime_profile = {
    'dt': [],
    'pt': [],
    'pn': []
}

batchSize = args.posebatch

_input_size = cfg.DATA_PRESET.IMAGE_SIZE
_output_size = cfg.DATA_PRESET.HEATMAP_SIZE
hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
_sigma = cfg.DATA_PRESET.SIGMA
norm_type = cfg.LOSS.get('NORM_TYPE', None)

if cfg.DATA_PRESET.TYPE == 'simple':
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    transformation = SimpleTransform(
        pose_dataset, scale_factor=0,
        input_size=_input_size,
        output_size=_output_size,
        rot=0, sigma=_sigma,
        train=False, add_dpg=False, gpu_device=args.device)

if __name__ == "__main__":

    eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(args.video)
    width  = int(cap.get(3)) # float `width`
    height = int(cap.get(4))  # float `height`

     
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    Frame_Number = -1

    video_save_opt = {
    'savepath': args.output_video_name,
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (width, height)
    }
    writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=args.qsize).start()
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    # Read until video is completed
    while(cap.isOpened()):
        Frame_Number += 1
        print(Frame_Number)
        if Frame_Number>=20:
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            results = model(frame.copy(),verbose=False)  # predict on an image
            orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #orig_img = torch.tensor(orig_img)
            #print(orig_img.shape)
            boxes = torch.reshape((results[0].boxes.xyxy).cpu(), (len(list(results[0].boxes.xyxy)),4))
            scores = torch.reshape((results[0].boxes.conf).cpu(), (len(list(results[0].boxes.xyxy)),1))
            ids = torch.zeros(len(list(results[0].boxes.xyxy)), 1)
            inps = torch.zeros(len(list(results[0].boxes.xyxy)), 3, 256, 192)
            cropped_boxes = torch.zeros(len(list(results[0].boxes.xyxy)), 4)
            for i, box in enumerate(boxes):
                inps[i], cropped_box = transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)
            if boxes is None or boxes.nelement() == 0:
                #writer.save(None, None, None, None, None, orig_img, str(Frame_Number))
                continue
            # Pose Estimation
            inps = inps.to(args.device)
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                if args.flip:
                    inps_j = torch.cat((inps_j, flip(inps_j)))
                hm_j = pose_model(inps_j)
                if args.flip:
                    hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                    hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                hm.append(hm_j)
            hm = torch.cat(hm)
            hm = hm.detach().cpu()


            writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, str(Frame_Number))   
        # Break the loop
        else: 
            break
     
    # When everything done, release the video capture object
    cap.release()
    while(writer.running()):
        time.sleep(1000)
        print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
    writer.stop()
    writer.terminate()
    writer.clear_queues()
     
    # Closes all the frames
    cv2.destroyAllWindows()