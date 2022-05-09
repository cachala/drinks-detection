# -*- coding: utf-8 -*-

import torch
import utils
import dataloader
import model
import label_utils
from torchvision import transforms
from engine import evaluate

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset has four classes 
    num_classes = 4

    path_test = "./drinks/labels_test.csv"
    test_dict = label_utils.build_label_dictionary(path_test)
    # use our dataset 
    dataset_test = dataloader.ImageDataset(test_dict, transforms.ToTensor())
    # define testing data loaders
    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                   batch_size=1, 
                                                   shuffle=False, 
                                                   num_workers=2,
                                                  collate_fn=utils.collate_fn)

    # get the model
    model = model.get_model_instance_segmentation(num_classes)
    # move model to the right device
    model.to(device)
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
