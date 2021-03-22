import torchvision
import torch
import pickle
import matplotlib.pyplot as plt
import os
import json
from dataset_management.task_creation import construct_tasks
from dataset_management.dataset_creation import TinyImagenetTask
from dataset_management.dataset_creation import get_cl_dset
from dataset_management.dataset_creation import get_task
from Trainer import Trainer


def make_datasets(cl_dset, datapath, randomise = False, show_dset_img_sample = False, premade_tasks = None):
    all_classes = list()
    for task_no in range(cl_dset['meta']['task_no']):
        task = get_task(cl_dset, task_no, datapath, verbose=True) if premade_tasks is None else premade_tasks[
            task_no]  # list of classes (strings) corresponding to task
        all_classes += task
    if randomise:
        from random import shuffle
        shuffle(all_classes)
    datasets = list()
    all_tasks_gathered = []
    for task_no in range(cl_dset['meta']['task_no']):
        # Getting task
        task = all_classes[task_no * cl_dset['meta']['cls_no']:(task_no + 1) * cl_dset['meta']['cls_no']]
        all_tasks_gathered.append(task)
        # Printing stuff
        print(f"task {task_no}{' shuffled' if randomise else ' unshuffled'}: {task}")
        # Make task dataset
        dset_task = TinyImagenetTask(os.path.join(datapath, "train"), task,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Resize((80, 80)),
                                         torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                         torchvision.transforms.RandomVerticalFlip(p=0.5),
                                         torchvision.transforms.RandomRotation((-90, 90)),  # min and max degrees
                                         torchvision.transforms.GaussianBlur(kernel_size=(1, 1), sigma=(0.1, 2.0)),
                                         torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.25),
                                                                              ratio=(0.3, 3.3)),
                                         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])
                                     ])
                                     )
        datasets.append(dset_task)
        im, label = dset_task[0]
        if show_dset_img_sample:
            # (task loaded, showing 10 imgs from task)
            print(f"\n\n Showing 10 images from the dataset of task {task_no}.")
            print(dset_task._subset)
            fig = plt.figure()
            axes = fig.subplots(2, 5)
            for i in range(10):
                im, label = dset_task[i]
                axes[i // 5, i % 5].set_title(f"label: {label}")
                axes[i // 5, i % 5].imshow(im.permute(1, 2, 0))
            fig.set_figheight(5)
            fig.set_figwidth(5)
            fig.suptitle(f"Dataset {task_no}")
            fig.tight_layout()
            plt.show(block=False)
            print("\n\n")
    return datasets, all_tasks_gathered

if False:
    # calculate test confusions using models gotten back
    epoch = 20
    model_types = ["Resnet18", "VGG11"]
    import os

    load_bpath = os.path.join("F:\projects", "robust-features-in-cl")
    test_loaders = []
    premade_tasks = [['n02107142', 'n02100583', 'n02106662'],
                     ['n04330267', 'n02101006', 'n02108551'],
                     ['n02099267', 'n02109961', 'n02099601'],
                     ['n03599486', 'n02108089', 'n04118538'],
                     ['n02109047', 'n02100735', 'n02802426'],
                     ['n02101388', 'n03179701', 'n02106166'],
                     ['n04209133', 'n02100877', 'n02958343']]  # the ones used in actual training, recovered from log
    fixed_shuffled_dsets = make_datasets(cl_dset, datapath, premade_tasks=premade_tasks)  # !!DO THIS
    shuff_test_loaders = []
    for task in range(cl_dset['meta']['task_no']):
        for model_type in model_types:
            _, test_dsload_task = torch.utils.data.random_split(datasets[task], [int(len(datasets[task]) * 0.8),
                                                                                 len(datasets[task]) - int(
                                                                                     len(datasets[task]) * 0.8)])
            test_loaders.append(
                torch.utils.data.DataLoader(test_dsload_task, batch_size=trainer.batch_size, shuffle=True,
                                            drop_last=False))
            _, shuff_test_dsload_task = torch.utils.data.random_split(fixed_shuffled_dsets[task],
                                                                      [int(len(fixed_shuffled_dsets[task]) * 0.8),
                                                                       len(fixed_shuffled_dsets[task]) - int(
                                                                           len(fixed_shuffled_dsets[task]) * 0.8)])
            shuff_test_loaders.append(
                torch.utils.data.DataLoader(shuff_test_dsload_task, batch_size=trainer.batch_size, shuffle=True,
                                            drop_last=False))


            def wrapper(test_loaders, suffix):
                trainer.set_model(model_type, cl_dset, load=True,
                                  load_attr={"load_bpath": load_bpath, "path": "savedump", "suffix": suffix,
                                             "num_epochs": epoch, "task": task})
                import pickle
                with open(os.path.join(basepath, 'savedump',
                                       f"{trainer.model.__class__.__name__}_{epoch}_epochs_metrics_task_{str(trainer.task_no) + trainer.save_appendix}"),
                          'rb') as filehandle:
                    old_metrics = pickle.load(filehandle)
                    old_metrics['test_confusions'] = [[] for _ in range(task + 1)]
                    old_metrics['test_entropies'] = [[] for _ in range(task + 1)]
                for prev_task in range(task + 1):
                    _, _, conf_task, entrop_task = trainer.test_epoch(test_loaders[prev_task])
                    old_metrics['test_confusions'][prev_task].append(conf_task)
                    old_metrics['test_entropies'][prev_task].append(entrop_task)
                trainer.save_metrics(metrics=old_metrics, epoch=epoch)


            wrapper(test_loaders, suffix="_imagenet" + ("_cl_t7_c3" if "VGG11" in model_type else ""))
            if model_type == "VGG11":
                wrapper(shuff_test_loaders, suffix="_imagenet_cl_t7_c3_shuffled")
            if model_type == "ResNet18":
                wrapper(test_loaders, suffix="_imagenet_ewc")

"""###Training for each task, all in a loop over tasks
Training on separate tasks:
Structure of metrics dictionary:
four entries (train & test loss, train & test acc),
each a list of lists:
for task i, the list contains all losses on the previous tasks including the current one for each epoch.
"""


def train_datasets(trainer, datasets, ewc=False, spec_decoup = False, save_appendix="", start_from_task=0, save_at_least_end_task = False):
    save_appendix += "_ewc" if ewc else ""
    save_appendix += "_specdec" if spec_decoup else ""
    print(f"---------------------Training {save_appendix}-----------------------------")
    trainer.set_saveloc(save_appendix)
    metrics = list()
    train_loaders, test_loaders = [], []
    fisher_diag = None
    prev_params = None
    for task_no in range(len(datasets)):
        trainer.set_task(task_no)
        dset_task = datasets[task_no]
        # split dataset for this task
        train_dset_task, test_dset_task = torch.utils.data.random_split(dset_task, [int(len(dset_task) * 0.8),
                                                                                    len(dset_task) - int(
                                                                                        len(dset_task) * 0.8)])
        print(f"        --------------- task {task_no} --------------- ")
        # add data loaders corresponding to current task
        train_loaders.append(
            torch.utils.data.DataLoader(train_dset_task, batch_size=trainer.batch_size, shuffle=True, drop_last=False))
        test_loaders.append(
            torch.utils.data.DataLoader(test_dset_task, batch_size=trainer.batch_size, shuffle=True, drop_last=False))

        # training (all tasks have the same # epochs and batch sizes)

        if task_no >= start_from_task:
            metrics_task, fisher_diag_task = trainer.train(train_loaders, test_loaders, spec_decoup = spec_decoup, ewc=ewc,
                                                           prev_fisher=fisher_diag, prev_params=prev_params)
            metrics.append(metrics_task)
            if ewc:
                # modify fisher diag to also reflect task that was trained on.
                if fisher_diag is None:
                    fisher_diag = fisher_diag_task
                else:
                    assert set(fisher_diag.keys()) == set(
                        fisher_diag_task.keys()), f"The previous fisher diagonal and the one obtained from {task_no} are dicts not corresponding to same parameters!"
                    for k in fisher_diag.keys():
                        fisher_diag[k] += fisher_diag_task[k]

                # retain params after task is done training

                with torch.no_grad():
                    prev_params = dict()
                    for name, param in trainer.model.named_parameters():
                        if param.requires_grad:
                            prev_params[name] = param.detach().clone()
                            prev_params[name].requires_grad = False
            # Ensure the previous params don't get trained as well. Might not be necessary since they don't get included in the optimizer anyway in the trainer procedures.
        if save_at_least_end_task and task_no >= start_from_task:
            trainer.save_metrics(metrics[-1], str(trainer.num_epochs))
            trainer.save_model(str(trainer.num_epochs))

        # RESETTING OPTIMIZER FOR BEGINNING OF NEW TASK
        trainer.set_optimizer()
    return metrics


def main(cudano):
    # -------------------------initial setup------------------------------#
    # setup device (CPU/GPU)
    if torch.cuda.is_available():
        device = torch.device(int(cudano) if cudano is not None and int(cudano) < torch.cuda.device_count() else 0)
    else:
        device = torch.device('cpu')
    print(f"Working on device: {device}.")

    base_dataset_name = "imagenet"
    datapath = os.path.join("data", base_dataset_name)
    basepath = os.path.join("")

    #-------------------------creating tasks------------------------------#

    # construct_tasks(datapath, min_imgs = 500, max_imgs = 5000)


    #-------------------------making datasets------------------------------#

    """
    cl_dset: a dictionary containing
            -> key 'meta': dict with keys: cls_no (classes per task), task_no (# of tasks), misc (number of misc classes)
            -> other keys: the disjoint superclasses used in making of tasks (their number is the number of classes/task)
    """

    file_names = ["cl_t5_c8", "cl_t10_c13", "cl_t7_c16", "cl_t5_c26"]
    datasets_dict = {}
    for split_file_name in file_names:
        print(f"----------------------{split_file_name}--------------------------------------")
        cl_dset = get_cl_dset(os.path.join(datapath, split_file_name + ".txt"))
        datasets, _ = make_datasets(cl_dset, datapath, randomise = False)
        shuffled_datasets, shuffled_tasks = make_datasets(cl_dset, datapath, randomise = True)
        datasets_dict[split_file_name] = {
            "meta_dset": cl_dset,
            "unshuffled": datasets,
            "shuffled": shuffled_datasets
        }
        with open(f'latest_shuffled_tasks_{split_file_name}.txt', 'w') as filehandle:
            json.dump(shuffled_tasks, filehandle)
        print("\n\n\n")

    #-------------------------setting up model and trainer params------------------------------------#

    trainer_params = {
        "batch_size": 50,  # 1500 batches for dataset of 7500
        "num_epochs": 20,
        "learning_algo": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "device": device,
        "basepath": basepath,
        "sd_coef": 0.003,
        "ewc_coef": 400
    }

    trainer = Trainer(trainer_params)

    """Choose model to be used"""
    model_type = "Resnet18"  # @param ["LeNet", "LeNet5", "Resnet101", "Resnet50", "Resnet18", "Resnet152", "Densenet169", "Densenet201", "ModDenseNet", "VGG11", VGG13"]

    trainer.save() #set the intermediary save functionality to True (method .nosave() for reverting; default is True)

    for ewc in [False, True]:
        for spec_decoup in [False, True]:
            if not (ewc and spec_decoup): #only train each regularization method, not both
                for task_div_denominator in datasets_dict.keys():
                    trainer.set_model(model_type, datasets_dict[task_div_denominator]["meta_dset"])
                    #unshuffled
                    train_datasets(trainer, datasets_dict[task_div_denominator]["unshuffled"], spec_decoup=spec_decoup, ewc=ewc,
                                   save_appendix=f"_{base_dataset_name}_{task_div_denominator}")
                    #shuffled (base)
                    train_datasets(trainer, datasets_dict[task_div_denominator]["shuffled"], spec_decoup = spec_decoup, ewc = ewc,
                            save_appendix = f"_{base_dataset_name}_{task_div_denominator}" + "_shuffled")


if __name__ == "__main__":
    import argparse
    import re
    parser = argparse.ArgumentParser()
    def regex_num(arg_value, pat = re.compile(r"^[0-9]+$")): #accept only one or more digits
        if not pat.match(arg_value):
            raise argparse.ArgumentTypeError
        return arg_value
    parser.add_argument("--cuda", help = "cuda core (just number)", type = regex_num)
    args = parser.parse_args()
    cudano = args.cuda
    main(cudano = cudano)