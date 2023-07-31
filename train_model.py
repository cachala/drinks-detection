# -*- coding: utf-8 -*-

import torch
import utils
import dataloader
import model
import label_utils
from torchvision import transforms
from engine import train_one_epoch, evaluate

if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset has four classes 
    num_classes = 4

    path_train = "./drinks/drinks/labels_train.csv"
    train_dict = label_utils.build_label_dictionary(path_train)
    # use our dataset 
    dataset = dataloader.ImageDataset(train_dict, transforms.ToTensor())
    # define training data loaders
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=2, 
                                              shuffle=True, 
                                              num_workers=2,
                                              collate_fn=utils.collate_fn)
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
   # print("TRAIN DICT", train_dict)
   # print("TEST DICT",test_dict)

    # get the model
    model = model.get_model_instance_segmentation(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # modify for the number of epochs
    num_epochs = 10

    for epoch in range(num_epochs):
      # train for one epoch, printing every 10 iterations
      train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
      # update the learning rate
      lr_scheduler.step()

      # evaluate on the test dataset
      evaluate(model, data_loader_test, device=device)

      torch.save(model, "saved_model.pth")
