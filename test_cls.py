import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import *
from models import generate_model
from lib.spatial_transforms import *

from data.data_loader import DataSet
from lib.utils import Logger
from cls import build_model
import time
import os,cv2
import sys
from PIL import Image

from lib.utils import AverageMeter, calculate_accuracy
from torch.autograd import Variable
from torch.optim import lr_scheduler
import eval_res

def get_label(res_tensor):
    res_numpy=res_tensor.data.cpu().numpy()
    labels=[]
    for row in res_numpy:
        labels.append(np.argmax(row))
    return labels

def deepSBD(video_path,temporal_length,model,spatial_transform,batch_size,**args):
    assert(os.path.exists(video_path))
    videocap=cv2.VideoCapture(video_path)
    status=True
    clip_batch=[]
    labels=[]
    image_clip=[]
    while status:
        for i in range(temporal_length-len(image_clip)):
            status,frame=videocap.read()
            if not status:
                break
            else:
                frame=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                frame=spatial_transform(frame)
                image_clip.append(frame)
        
        image_clip+=[image_clip[-1] for _ in range(temporal_length-len(image_clip))]

        if len(image_clip)==temporal_length:
            clip = torch.stack(image_clip, 0).permute(1, 0, 2, 3)
            clip_batch.append(clip)
            image_clip=image_clip[int(temporal_length/2):]
        
        if len(clip_batch)==batch_size or not status:
            clip_tensor=torch.stack(clip_batch, 0)
            clip_tensor=Variable(clip_tensor).cuda()
            results=model(clip_tensor)
            labels+=get_label(results)
            clip_batch=[]

    final_res=[]
    i=0
    while i<len(labels):
        if labels[i]>0:
            label=labels[i]
            begin=i
            i+=1
            while i<len(labels) and labels[i]==labels[i-1]:
                i+=1
            end=i-1
            final_res.append((begin*temporal_length/2+1,end*temporal_length/2+16+1,label))
        else:
            i+=1
    return final_res

def load_video_list(path):
    with open(path,'r') as f:
        return [line.strip('\n') for line in f.readlines()]

def load_checkpoint(model,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

def get_result(opt,model):
    spatial_transforms=get_test_spatial_transform(opt)
    video_list=load_video_list(opt.test_list_path)
    res={}

    for idx,videoname in enumerate(video_list):
        print("Process {} {}".format(idx,videoname),flush=True)
        labels=deepSBD(os.path.join(opt.root_dir,opt.test_subdir,videoname),opt.sample_duration,model,spatial_transforms,opt.batch_size)
        _res={'cut':[],'gradual':[]}
        for begin,end,label in labels:
            if label==2:
                _res['cut'].append((begin,end))
            else:
                _res['gradual'].append((begin,end))
        res[videoname]=_res
    return res


def test(opt,model):
    out_path=os.path.join(opt.result_dir,'results.json')
    if not os.path.exists(out_path):
        res=get_result(opt,model)
        json.dump(res,open(out_path,'w'))
    
    eval_res.eval(out_path,opt.gt_dir)

def main(opt):
    model = build_model(opt,"test")
    load_checkpoint(model,opt.weights)
    model.eval()
    test(opt,model)

if __name__ == '__main__':
    opt = parse_test_args()
    test(opt)


    
