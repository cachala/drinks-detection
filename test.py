# -*- coding: utf-8 -*-

import torch
import utils
import dataloader
import model
import label_utils
import numpy as np
from torchvision import transforms
from engine import evaluate
from label_utils import get_box_color, index2class
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset has four classes 
    num_classes = 4

    path_test = "./drinks/drinks/labels_test.csv"
    test_dict = label_utils.build_label_dictionary(path_test)
    # use our dataset 
    dataset_test = dataloader.ImageDataset(test_dict, transforms.ToTensor())
    # define testing data loaders
    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                   batch_size=1, 
                                                   shuffle=False, 
                                                   num_workers=2,
                                                  collate_fn=utils.collate_fn)

    model = torch.load("saved_model.pth")
    model.to(device)

    img, _ = dataset_test[28]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    print(prediction)

    box_coords = torch.as_tensor(prediction[0]['boxes'], device='cpu')
    labels = torch.as_tensor(prediction[0]['labels'], device='cpu')
    

    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    idx = 0
    for label in box_coords:
      # default label format is xmin, xmax, ymin, ymax
      w = label[2] - label[0]
      h = label[3] - label[1]
      x = label[0]
      y = label[1]
      category = int(labels[idx])
      color = get_box_color(category)
      classes = index2class(int(labels[idx]))
      rect = Rectangle((x, y),
                         w,
                         h,
                         linewidth=2,
                         edgecolor=color,
                         facecolor='none',
                         label='tryyy')
      ax.add_patch(rect)
      ax.annotate(classes, (x, label[3]+12), 
                          color=color, 
                          fontsize=7,
                          weight='bold')

      idx += 1
    plt.show()

    













