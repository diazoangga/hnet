import torch
from model.lanenet import hnet
from model.lanenet import hnet_loss
import numpy as np
import cv2
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, help='The phase is train or pretrain')
parser.add_argument('--pre_hnet_weights', type=str, help='The pre hnet weights path')
parser.add_argument('--hnet_weights', type=str, help='The hnet model weights path')
parser.add_argument('--bs', type=int, help='Batch size for HNet training')
args = parser.parse_args()

num_lanes = 4
net = hnet.HNet(num_lanes)
loss = hnet_loss.hnet_loss()

#def dataloader()