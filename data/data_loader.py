import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np
import random
import cv2

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def video_loader(video_path,video_dir_path, frame_indices,sample_duration,img=False):
    video = []
    try:
        if not img:
            video_cap=cv2.VideoCapture(video_path)
            video_cap.set(1,frame_indices)
            for i in range(sample_duration):
                status,frame=video_cap.read()
                if status:
                    frame=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                    video.append(frame)
                else:
                    break
            return video
    except Exception as e:
        pass

def get_default_video_loader():
    return video_loader


def make_dataset(root_path, image_list_path, sample_duration):
    image_list=[]
    with open(image_list_path, 'r') as f:
        for line in f.readlines():
            words=line.split(' ')
            begin=int(words[-2])
            info={"video_path":os.path.join(root_path,words[0]),"begin":begin,"label":int(words[-1])}
            assert(os.path.exists(info['video_path']))
            gts=[]
            for i in range(4,len(words),2):
                gts.append(((float(words[i])-begin)/sample_duration,(float(words[i+1])-begin)/sample_duration))
            info["gt_intervals"]=gts
            info["sample_duration"]=sample_duration
            image_list.append(info)

    return image_list


class DataSet(data.Dataset):

    def __init__(self, root_path, image_list_path,
                 spatial_transform=None, temporal_transform=None, target_transform=None,
                 sample_duration=16, get_loader=get_default_video_loader):
        self.image_list = make_dataset(root_path, image_list_path,sample_duration)
        print(len(self.image_list))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

        self.weights=self.make_weights()
    
    def make_weights(self):
        labels_cnt={}
        for img_info in self.image_list:
            label=img_info['label']
            if label not in labels_cnt:
                labels_cnt[label]=0
            labels_cnt[label]+=1
        weights=[]
        for img_info in self.image_list:
            label=img_info['label']
            weights.append(1.0/labels_cnt[label])
        return weights

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        video_path = self.image_list[index]['video_path']

        begin_indices = self.image_list[index]['begin']

        sample_duration=self.image_list[index]['sample_duration']

        clip = self.loader(video_path,None, begin_indices,sample_duration)
        raw_clip=clip

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            if len(clip)==0:
                return self.__getitem__(random.randint(0,len(self.image_list)-1))
            if len(clip)!=sample_duration:
                clip+=[clip[-1] for _ in range(sample_duration-len(clip))]
        assert(len(clip)==sample_duration)
            
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        gts=self.image_list[index]['gt_intervals']
        label = self.image_list[index]['label']

        target=label
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.image_list)

