import sys
import os
from glob import glob

from argparse import ArgumentParser
import numpy as np
import cv2 as cv
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A


class EyeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 48, 3)
        self.conv3 = nn.Conv2d(48, 128, 3)
        self.do = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.do(x)
        x = self.fc1(x)
        return x


ATFMS_CUSTOM_TEST = A.Compose([
    A.Normalize(mean=0.5, std=0.5),
    A.Resize(24, 24)
])


def clf_eyenet(net, img):
    img = ATFMS_CUSTOM_TEST(image=img)['image']
    inp_t = torch.tensor(img, dtype=torch.float).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        out = net(inp_t).detach().cpu().numpy().ravel()
    return softmax(out)[1] > 0.6


def main():
	path = sys.argv[1]
	
	print('Loading model...')
	net = EyeNet()
	net.load_state_dict(torch.load('./eyenet.pt'))
	net = net.eval()
	
	print('Loading *.jpg images from path = {}...'.format(path))
	
	result = []
	for fname in sorted(glob(os.path.join(path, '*.jpg'))):
		print(fname)
		try:
			img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
			label = clf_eyenet(net, img)
			result.append((fname, label))
		except KeyboardInterrupt:
			print('Interrupted')
			try:
				sys.exit(0)
			except SystemExit:
				os._exit(0)
		except:
			pass  # unknown error of any kind
		
	with open('./result.csv', 'w') as f:
		for line in result:
			f.write('{},{}\n'.format(line[0], int(line[1])))


main()
