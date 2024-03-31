import os
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir',required=True)
args = parser.parse_args()

list_ = ['train2017','val2017']


for fold in list_:
    os.makedirs('new_dir/images/'+fold,exist_ok=True)
    labels = glob.glob('new_dir/labels/'+fold+'/*')

    for lab in labels:
        name = os.path.basename(lab).split('.')[0]+'.jpg'
        os.system('cp '+ args.image_dir + name +' new_dir/images/'+fold+'/')