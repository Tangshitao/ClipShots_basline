import torch
from torch import nn

import models.resnet as resnet
from models.deepSBD import *
def generate_model(opt):
    assert opt.model in ['resnet', 'alexnet']

    if opt.model=='alexnet':
        model=deepSBD()
    elif opt.model == 'resnet':
        from models.resnet import get_fine_tuning_parameters

        model = resnet.resnet18(num_classes=opt.n_classes,
                                sample_size=opt.sample_size, sample_duration=opt.sample_duration)
    else:
        raise Exception("Unknown model name")

    return model
