# -*- coding: utf-8 -*-
"""Robust Features for CL.ipynb
Original file is located at
https://colab.research.google.com/drive/1BrQRoMzjRNyIdG-x0H73xHmK53ulcL_g
"""



import torchvision
import torch
import pickle
import matplotlib.pyplot as plt
import os

# setup device (CPU/GPU)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')

print(f"Working on device: {device}.")

dataset_name = "imagenet"
datapath = os.path.join("data", dataset_name)
basepath = os.path.join("")

"""Imgnet premade downloader: https://towardsdatascience.com/how-to-scrape-the-imagenet-f309e02de1f4"""

"""https://medium.com/@staticmukesh/how-i-managed-to-delete-1-billions-files-from-my-google-drive-account-9fdc67e6aaca """

# from dataset_management.task_creation import construct_tasks
# construct_tasks(datapath, min_imgs = 600, max_imgs = 5000)

"""## **Datasets** """

from dataset_management.dataset_creation import TinyImagenetTask
from dataset_management.dataset_creation import get_cl_dset

"""### Getting the datasets"""
# cl_dset = get_cl_dset(os.path.join(datapath, "cl_t7_c21.txt"))
# cl_dset = get_cl_dset(os.path.join(datapath, "cl_t5_c15.txt"))
"""
cl_dset: a dictionary containing
        -> key 'meta': dict with keys: cls_no (classes per task), task_no (# of tasks), misc (number of misc classes)
        -> other keys: the disjoint superclasses used in making of tasks (their number is the number of classes/task)
"""
split_file_name = "cl_t7_c3.txt"
cl_dset = get_cl_dset(os.path.join(datapath, split_file_name))

def make_datasets(cl_dset, datapath, randomise = False, show_dset_img_sample = False):
    from dataset_management.dataset_creation import get_task

    all_classes = list()
    for task_no in range(cl_dset['meta']['task_no']):
        print(f"Showing info for task {task_no}")
        task = get_task(cl_dset, task_no, datapath, verbose=True)  # list of classes corresponding to task
        all_classes += task
    if randomise:
        from random import shuffle
        shuffle(all_classes)
    datasets = list()
    for task_no in range(cl_dset['meta']['task_no']):
        #Getting task
        task = all_classes[task_no * cl_dset['meta']['cls_no']:(task_no + 1) * cl_dset['meta']['cls_no']]

        #Printing stuff
        print(f"Classes in task {task_no}: {task}")
        task_img_counts = [len([img for img in os.listdir(os.path.join(datapath, "train", f"{cls_id}"))]) for cls_id in task]
        print(f"Task {task_no} has {sum(task_img_counts)} images in total, proportioned: {[ ( cls_cnt, '{:.2%}'.format(cls_cnt/float(sum(task_img_counts))) ) for cls_cnt in task_img_counts]} .\n")

        #Make task dataset
        dset_task = TinyImagenetTask(os.path.join(datapath, "train"), task,
                            transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Resize((60,60)),
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
        if show_dset_img_sample:
            #(task loaded, showing 10 imgs from task)
            print(f"\n\n Showing 10 images from the dataset of task {task_no}.")
            print(dset_task._subset)
            fig = plt.figure()
            axes = fig.subplots(2, 5)
            for i in range(10):
              im, label = dset_task[i]
              axes[i//5, i%5].set_title(f"label: {label}")
              axes[i//5, i%5].imshow(im.permute(1,2,0))
            fig.set_figheight(5)
            fig.set_figwidth(5)
            fig.suptitle(f"Dataset {task_no}")
            fig.tight_layout()
            plt.show(block = False)
            print("\n\n")
    return datasets


datasets = make_datasets(cl_dset, datapath, randomise = False)
# baseline_datasets = make_datasets(cl_dset, datapath, randomise = True)


##Training



trainer_params = {
    "batch_size": 15, #1500 batches for dataset of 7500
    "num_epochs": 1,
    "learning_algo": "adam",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "device": device,
    "basepath": basepath
}

from Trainer import Trainer

trainer = Trainer(trainer_params)

"""Choose model to be used"""
model_type = "Resnet18" #@param ["LeNet", "LeNet5", "Resnet101", "Resnet50", "Resnet18", "Resnet152", "Densenet169", "Densenet201", "ModDenseNet"]

get_back = False
trainer.set_model(model_type, cl_dset, load = get_back, load_attr = {"basepath": basepath, "path": "savedump", "suffix": "_imagenet", "num_epochs": trainer.num_epochs, "task": len(datasets)-1} )

"""###Training for each task, all in a loop over tasks

Training on separate tasks:

Structure of metrics dictionary:
four entries (train & test loss, train & test acc),
each a list of lists:
for task i, the list contains all losses on the previous tasks including the current one for each epoch.
"""

def train_datasets(trainer, datasets, ewc = False, save_appendix = "", start_from_task = 0, save_at_least_end_task = True):
    metrics = list()
    train_loaders, test_loaders = [], []
    fisher_diag = None
    prev_params = None
    trainer.set_saveloc(save_appendix)
    for task_no in range(len(datasets)):
        trainer.set_task(task_no)
        dset_task = datasets[task_no]
        #split dataset for this task
        train_dset_task, test_dset_task = torch.utils.data.random_split(dset_task,[int(len(dset_task)*0.8),len(dset_task) - int(len(dset_task)*0.8)])

        #add data loaders corresponding to current task
        train_loaders.append(torch.utils.data.DataLoader(train_dset_task, batch_size=trainer.batch_size, shuffle=True, drop_last=False))
        test_loaders.append(torch.utils.data.DataLoader(test_dset_task, batch_size=trainer.batch_size, shuffle=True, drop_last=False))

        #training (all tasks have the same # epochs and batch sizes)

        if task_no >= start_from_task:
            metrics_task, fisher_diag_task = trainer.train(train_loaders, test_loaders, prev_fisher = fisher_diag, prev_params = prev_params)
            metrics.append(metrics_task)
            if ewc:
                # modify fisher diag to also reflect task that was trained on.
                if fisher_diag is None:
                    fisher_diag = fisher_diag_task
                else:
                    fisher_diag += fisher_diag_task

                # retain params after task is done training

                with torch.no_grad():
                    for param in trainer.model.parameters():
                        if prev_params is None:
                            prev_params = param.detach().clone().view([1, -1])
                        else:
                            prev_params = torch.cat((prev_params, param.detach().clone().view([1, -1])), dim=1)
            #Ensure the previous params don't get trained as well. Might not be necessary since they don't get included in the optimizer anyway in the trainer procedures.
            prev_params.requires_grad = False
        if save_at_least_end_task and task_no >= start_from_task:
            trainer.save_metrics(metrics[-1], str(trainer.num_epochs))
            trainer.save_model(str(trainer.num_epochs))

        #RESETTING OPTIMIZER FOR BEGINNING OF NEW TASK
        trainer.set_optimizer()
    return metrics


if True:
    trainer.save()
    metrics = train_datasets(trainer, datasets, ewc = True, save_appendix = f"_{dataset_name}", start_from_task = 0)

    """Dumping overall metrics (kinda useless, consider eliminating)"""
    with open(os.path.join(basepath,'savedump',
        f"{model.__class__.__name__}_{str(trainer.num_epochs)}_epochs_lr{str(trainer.lr)}_overall_metrics_{dataset_name}"), 'wb') as filehandle:
          pickle.dump(metrics, filehandle)

    """Training on baseline dataset"""

    #@title Train on baseline (reinitialise model if it was already trained on nonbaseline tasks)
    reinitialise_model = False  #@param {type:"boolean"}
    if reinitialise_model:
      trainer.set_model(model_type, cl_dset)
    #MODEL & TRAINER MUST BE REINITIALISED IF ALREADY TRAINED ON GOOD TASKS
    """
    metrics_baseline = train_datasets(trainer, basepath, baseline_datasets, save_on_the_way = True, save_appendix = f"_baseline_{dataset_name}")
    with open(os.path.join(basepath, "savedump",
                           f"{model.__class__.__name__}_{str(trainer.num_epochs)}_epochs_lr{str(trainer.lr)}_overall_metrics_baseline_{dataset_name}")
            , 'wb') as filehandle:
          pickle.dump(metrics_baseline, filehandle)
    """

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

