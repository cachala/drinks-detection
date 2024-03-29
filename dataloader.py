# -*- coding: utf-8 -*-
"""dataloader.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-yecXvBp5i7RN-ZX80rbL_k4ah0wNEJC
"""

import torch
import torchvision
from torchvision import transforms
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, dictionary, transform=None):
        self.dictionary = dictionary
        self.transform = transform

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # retrieve the image filename
        key = list(self.dictionary.keys())[idx]
         # open the file as a PIL image
        img = Image.open(key)

        # retrieve all labels
        boxes = []
        labels = []
        area = []
        for i in range(len(self.dictionary[key])):
            boxes.append(self.dictionary[key][i][0])
            labels.append(self.dictionary[key][i][1])
            area.append(self.dictionary[key][i][2])
    
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.as_tensor(idx, dtype=torch.int64)
        area = torch.tensor(area)
        # suppose all instances are not crowd
        iscrowd = torch.zeros(len(self.dictionary[key]), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform:
            img = self.transform(img)
        # return a list of images and corresponding labels
        return img, target
