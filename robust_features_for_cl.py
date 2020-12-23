# -*- coding: utf-8 -*-
"""Robust Features for CL.ipynb
Original file is located at
https://colab.research.google.com/drive/1BrQRoMzjRNyIdG-x0H73xHmK53ulcL_g
"""



import torchvision
import torch
import pickle

import matplotlib.pyplot as plt

# setup device (CPU/GPU)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')

print(f"Working on device: {device}.")
basepath="./"

"""Imgnet premade downloader: https://towardsdatascience.com/how-to-scrape-the-imagenet-f309e02de1f4"""

"""https://medium.com/@staticmukesh/how-i-managed-to-delete-1-billions-files-from-my-google-drive-account-9fdc67e6aaca """

# from dataset_management.tinyimgnet.tiny_imagenet_stats import tiny_imagenet_stats
# tiny_imagenet_stats(basepath)

"""## **Datasets** """

from dataset_management.tinyimgnet.datasets import get_cl_dset
from dataset_management.tinyimgnet.datasets import get_task
from dataset_management.tinyimgnet.datasets import TinyImagenetTask

"""### Getting the datasets"""
cl_dset = get_cl_dset(basepath+"cl_t5_c15.txt")
"""
cl_dset: a dictionary containing
        -> key 'meta': dict with keys: cls_no (classes per task), task_no (# of tasks), misc (number of misc classes)
        -> other keys: the disjoint superclasses used in making of tasks (their number is the number of classes/task)
"""

datasets = list()
all_classes = list()
for task_no in range(cl_dset['meta']['task_no']):
  print(f"Showing info for task {task_no}")
  task = get_task(cl_dset, task_no, basepath, verbose=False) #list of classes corresponding to task
  all_classes += task
  dset_task = TinyImagenetTask(basepath+"/data/tiny-imagenet-200/train", task,
                        transform = torchvision.transforms.Compose([
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.RandomHorizontalFlip(p=0.5),
                          torchvision.transforms.RandomVerticalFlip(p=0.5),
                          torchvision.transforms.RandomRotation((-90,90)), #min and max degrees
                          torchvision.transforms.GaussianBlur(kernel_size = (1,1), sigma=(0.1, 2.0)),
                          torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
                          torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
                        )
  datasets.append(dset_task)
  im, label = dset_task[0]
  if task_no == 0:
    print(f"We have images of shape {im.shape}")
  show_dataset_task_sample = False
  if show_dataset_task_sample:
    #(task loaded, showing 10 imgs from task)
    print(f"\n\n Showing 10 images from the dataset of task {task_no}.")
    print(dset_task._subset)
    fig = plt.figure()
    axes = fig.subplots(2, 5)
    for i in range(10):
      im, label = dset_task[500*i] #each class has 500 images
      axes[i//5, i%5].set_title(f"label: {label}")
      axes[i//5, i%5].imshow(im.permute(1,2,0))
    fig.set_figheight(5)
    fig.set_figwidth(5)
    fig.suptitle(f"Dataset {task_no}")
    fig.tight_layout()
    fig.show()
    print("\n\n")

# import sys
# sys.exit()

from random import shuffle
shuffle(all_classes)

baseline_datasets = list()
for task_no in range(cl_dset['meta']['task_no']):
  print(f"Showing info for task {task_no}")
  subset = all_classes[task_no*cl_dset['meta']['cls_no']:(task_no+1)*cl_dset['meta']['cls_no']]
  dset_task = TinyImagenetTask(basepath+"data/tiny-imagenet-200/train", task,
                        transform = torchvision.transforms.Compose([
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.RandomHorizontalFlip(p=0.5),
                          torchvision.transforms.RandomVerticalFlip(p=0.5),
                          torchvision.transforms.RandomRotation((-90,90)), #min and max degrees
                          torchvision.transforms.GaussianBlur(kernel_size = (1,1), sigma=(0.1, 2.0)),
                          torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
                          torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
                        )
  baseline_datasets.append(dset_task)

"""A bit of dataset analysis"""

# print(len(datasets[1]))

#@title Mis dset analysis

# image_count_per_task_then_class = []
# for task_no in range(cl_dset['meta']['task_no']):
#     image_count_per_task_then_class.append([])
#     for sclass_label in range(cl_dset['meta']['cls_no']):
#       image_count_per_task_then_class[-1].append(
#           sum(  [1 if datasets[task_no][i][1]==sclass_label else 0 for i in range(len(datasets[task_no]))]  ) 
#       )
#       print(f"On task {task_no} superclass {sclass_label} has {image_count_per_task_then_class[-1][-1]} representatives.")

# import plotly.graph_objects as go

# fig = go.Figure(data=[go.Table(
#     header=dict(values
#                 = ['Superclass label']+[f'Task {i}' for i in range(cl_dset['meta']['task_no'])]),
#       cells=dict(values = [ [i for i in range(cl_dset['meta']['cls_no'])] ] +
#                image_count_per_task_then_class)
#                 )])
# #     cells=dict(values = [ [i for i in range(cl_dset['meta']['cls_no'])] ] +
# #                [
# #                 [ sum(  [1 if datasets[task_no][i][1]==sclass_label else 0 for i in range(len(datasets[task_no]))]  ) 
# #                for sclass_label in range(cl_dset['meta']['cls_no']) ] 
# #                 for task_no in range(cl_dset['meta']['task_no']) 
# #                 ])
# #                 )])
# fig.show()




##Training

from Models import LeNet
from Models import LeNet5
from Models import ModDenseNet
from Models import Block
from Trainer import Trainer

"""Choose model to be used"""

def get_model(model_type):
  if model_type == "LeNet":
    model = LeNet()
  elif model_type == "LeNet5":
    model = LeNet5()
  elif model_type == "ModDenseNet":
    model = ModDenseNet(block = Block, image_channels = 3, num_classes = cl_dset['meta']['cls_no'], device = device)
  elif "Resnet" in model_type:
    if model_type == "Resnet152":
      model = torchvision.models.resnet152(pretrained=False)
    elif model_type == "Resnet101":
      model = torchvision.models.resnet101(pretrained=False)
    elif model_type == "Resnet50":
      model = torchvision.models.resnet50(pretrained=False)
    elif model_type == "Resnet18":
      model = torchvision.models.resnet18(pretrained=False)
    elif model_type == "Densenet169":
      model = torchvision.models.densenet169(pretrained=False)
    elif model_type == "Densenet201":
      model = torchvision.models.densenet201(pretrained=False)
    model.fc = torch.nn.Sequential(
              torch.nn.Linear(in_features = model.fc.in_features, out_features = model.fc.in_features//4, bias = True), #was initially /2 -> /4 -> /8
              torch.nn.ReLU(),
              torch.nn.Linear(in_features = model.fc.in_features//4, out_features = model.fc.in_features//8, bias = True), 
              torch.nn.ReLU(),
              torch.nn.Linear(in_features = model.fc.in_features//8, out_features = cl_dset['meta']['cls_no'], bias = True)
            )
  return model

#@title Choose model
"""Getting info back"""


model_type = "ModDenseNet" #@param ["LeNet", "LeNet5", "Resnet101", "Resnet50", "Resnet18", "Resnet152", "Densenet169", "Densenet201", "ModDenseNet"]
model = get_model(model_type)

get_back = False
if get_back:
    model.load_state_dict(basepath+"savedump/" + model.__class__.__name__ + '_' + str(trainer.num_epochs) + '_epochs' + '_lr' + str(
        trainer.lr) + '_model_after_task' + str(task_no))


#auxiliary

hyperparams = {
    "batch_size": 5, #1500 batches for dataset of 7500
    "num_epochs": 25,
    "learning_algo": "adam",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "device": device
}

trainer = Trainer(model, hyperparams)

"""###Training for each task, all in a loop over tasks

Training on separate tasks:

Structure of metrics dictionary:
four entries (train & test loss, train & test acc),
each a list of lists:
for task i, the list contains all losses on the previous tasks including the current one for each epoch.
"""

def train_datasets(model, trainer, datasets, save_on_the_way = True, save_appendix = "", start_from_task = 0):
  metrics = list()
  train_loaders, test_loaders = [], []
  for task_no in range(len(datasets)):
    dset_task = datasets[task_no]

    #split dataset for this task
    train_dset_task, test_dset_task = torch.utils.data.random_split(dset_task, [int(len(dset_task)*0.8),int(len(dset_task)*0.2)])

    #add data loaders corresponding to current task
    train_loaders.append(torch.utils.data.DataLoader(train_dset_task, batch_size=trainer.batch_size, shuffle=True, drop_last=False))
    test_loaders.append(torch.utils.data.DataLoader(test_dset_task, batch_size=trainer.batch_size, shuffle=True, drop_last=False))
    
    print(f"\n \n Finished processing dataset for task {task_no}. Proceeding to training.")
    #training (all tasks have the same # epochs and batch sizes)
    if task_no >= start_from_task:
        metrics.append(trainer.train(train_loaders, test_loaders))
    if save_on_the_way and task_no >= start_from_task:
      import pickle
      with open(basepath+'savedump'+model.__class__.__name__+'_'+str(trainer.num_epochs)+'_epochs'+'_lr'+str(trainer.lr)+'_metrics_task_'+str(task_no)+save_appendix, 'wb') as filehandle:
        pickle.dump(metrics[-1], filehandle)
      torch.save(model.state_dict(), 
                basepath+'savedump'+model.__class__.__name__+'_'+str(trainer.num_epochs)+'_epochs'+'_lr'+str(trainer.lr)+'_model_after_task'+str(task_no)+save_appendix)
    # plot_metrics(metrics[task_no], title = f"Task {task_no} metrics after {trainer.num_epochs} epochs with learning rate {trainer.lr} for model {model.__class__.__name__}.")
    #RESETTING OPTIMIZER FOR BEGINNING OF NEW TASK
    if hyperparams['learning_algo'] == 'adam':
            trainer.optimizer = torch.optim.Adam(params = trainer.model.parameters(),
                                  lr = hyperparams['learning_rate'], weight_decay = hyperparams['weight_decay'])
    else:
            trainer.optimizer = torch.optim.SGD(params = trainer.model.parameters(), 
                                 lr = hyperparams['learning_rate'], weight_decay = hyperparams['weight_decay'])
  return metrics

metrics = train_datasets(model, trainer, datasets, save_on_the_way = True, save_appendix = "", start_from_task = 1)
with open(basepath+'savedump/'+model.__class__.__name__+'_'+str(trainer.num_epochs)+'_epochs'+'_lr'+str(trainer.lr)+'_overall_metrics', 'wb') as filehandle:
      pickle.dump(metrics, filehandle)

"""Training on baseline dataset"""

#@title Train on baseline (reinitialise model if it was already trained on nonbaseline tasks)
reinitialise_model = False #@param {type:"boolean"}
if reinitialise_model:
  model = get_model(model_type)
#MODEL & TRAINER MUST BE REINITIALISED IF ALREADY TRAINED ON GOOD TASKS
metrics_baseline = train_datasets(model, trainer, baseline_datasets, save_on_the_way = True, save_appendix = "_baseline")
with open(basepath+"savedump/"+model.__class__.__name__+'_'+str(trainer.num_epochs)+'_epochs'+'_lr'+str(trainer.lr)+'_overall_metrics'+"_baseline", 'wb') as filehandle:
      pickle.dump(metrics_baseline, filehandle)

"""Training on big dataset"""
#NOT NEEDED ANYMORE
# def train_big_dset(model, trainer, datasets):#now train on a the whole dataset, of all tasks:
#   big_dset = torch.utils.data.ConcatDataset(datasets)
#   train_dset, test_dset = torch.utils.data.random_split(
#       big_dset, [int(len(big_dset)*0.8), int(len(big_dset)*0.2)])
#   #create data loaders
#   train_loader = [torch.utils.data.DataLoader(train_dset, batch_size=trainer.batch_size, shuffle=True, drop_last=False)] #must be list (with one element in this case)
#   test_loader = [torch.utils.data.DataLoader(test_dset, batch_size=trainer.batch_size, shuffle=True, drop_last=False)]
#   metrics_big_dset = trainer.train(train_loader, test_loader)
#   import pickle
#   with open(basepath+"savedump/"+model.__class__.__name__+'_'+str(trainer.num_epochs)+'_epochs'+'_lr'+str(trainer.lr)+'_metrics_big_dset', 'wb') as filehandle:
#     pickle.dump(metrics_big_dset, filehandle)
#   torch.save(model.state_dict(),
#               basepath+"savedump/"+model.__class__.__name__+'_'+str(trainer.num_epochs)+'_epochs'+'_lr'+str(trainer.lr)+'_model_after_big_dset')
#
#   plot_metrics(metrics_big_dset, title = f"Metrics for combined dataset for model {model.__class__.__name__}.")

from plotting_utils import plot_accuracies_for_model
