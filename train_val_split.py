"""
this module can be used to split data which are in imagefolder format
i.e:
    format: root/class1/img1.jpg,
            root/class1/img2.jpg,
            root/class2/img1.jpg,
            root/class3/img20.jpg,
"""

import os
import shutil
import numpy as np

def copy_files(src, files, dst):
    file_locs = [os.path.join(src, i) for i in files]

    for file_loc in file_locs:
        shutil.copy(file_loc, file_loc.replace(root, dst))

def split(root, save_dir, split_ratio):
    save_dir = os.path.expanduser(save_dir)


    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    train_val_dirs = {i: os.path.join(save_dir, i) for i in ["train", "val"]}

    for key, val in train_val_dirs.items():
        if not os.path.exists(val):
            os.mkdir(val)
        
    dirs = os.listdir(root)

    res_dir = np.array([[ os.path.join(val,dir) for dir in dirs] for _, val in  train_val_dirs.items()])
    
    res_dir = res_dir.flatten()

    for i in res_dir:
        if not os.path.exists(i):
            os.mkdir(i)

    for dir, _, files in os.walk(root):
        if files :
            total_count = len(files)
            split_index = int(split_ratio*total_count)
            copy_files(dir, files[:split_index], train_val_dirs["train"])
            copy_files(dir, files[split_index:], train_val_dirs["val"])

def total_count(root):
    count = 0
    for dir, _, files in os.walk(root):
        count += len(files)
    print(count)


def total_classes(root):
    count = 0
    files = os.listdir(root)
    for file in files:
        if not os.path.isfile(file):
            count += 1
    print(count)
        

if __name__ == "__main__":
    root = "/nfs/71_datasets/oralcareProject/classification/oralcare_cls_v0.1/oralcare_data_v0.1/colgate/data"
    print(total_classes(root), total_count(root))
    split_ratio = 0.9
    save_dir  = "./colgate_data"
    split(root, save_dir, split_ratio)
