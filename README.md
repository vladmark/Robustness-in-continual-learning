# Robust Features for Continual Learning

## Overview
In this project, we examine the possibility that semantic segmentation of an image database like ImageNet might mitigate the phenomenon of catastropic forgetting on simple successive classification tasks.

The goal is to learn a series of several image classification tasks (in our experiments, either 5 or 7 tasks) successively, where training on task $k$ is stopped once overfitting starts or good performance is reached. Once learning of task $k+1$ begins, there is no revisiting of images from tasks $1, \ldots, k$. The goal is to maintain good performance on tasks $1, \ldots, k$ while learning task $k+1$.

For each experiment, the number of classes to be learned supervisedly is fixed and is the same number $n_c$ for all tasks. Say we have $n_t$ tasks. The idea of semantic segmentation is that for each $i$ from $1$ to $n_c$, the classes $\{ c_i^1, \ldots, \ldots, c_i^{n_t} \}$ are semantically correlated, not just randomly chosen. For example, if in task 1 we learn "dog" as the first class, we might learn "cat" as the first class in task 2. Thus, the experiment becomes as much about catastrophic forgetting as about the amount of semantic information that is present in the images themselves.


## Algorithm description
Firstly, we take care of the problem of finding collections of tasks that are large enough in both the number of tasks and in the number of classes per task. For this, we use the semantic graph of ImageNet, where we keep for each node only its parent with the most number of children. Then we do a depth-first search to find the best candidates for "superclasses" i.e. collections of semantically correlated classes that can be used across different tasks, given several sets of constraints on both the number of classes per tasks and the number of tasks.

Next we use a variety of models and hyperparameters to train the different tasks, corresponding to different dataloaders using the Torch ImageFolder and DataLoader classes to handle dataset management and batch splitting, respectively. The models we mainly focus on are Resnet18, ResNet50 and VGG11 from the Torchvision library, as well as a custom LeNet model. We use both SGD and Adam as optimizers, different learning rates and a weight decay of $1e-5$.

As a further step, we also test the difference that two other methods for mitigating catastrophic forgetting have combined with our approach. The first is a technique known as ["spectral decoupling" (SD)](https://arxiv.org/pdf/2103.17171.pdf), which in our code amounts to penalising the size of the logits that the model produces in training - in other words, to penalizing overconfidence. The second technique is knonw as ["elastic weight consolidation" (EWC)](https://arxiv.org/pdf/1612.00796.pdf) and makes use of the Fisher information matrix to provide an additional cost for changing each parameter in the model; this is determined by how relevant the parameter is to performance on the previous task, and this relevance is measured by the Fisher information matrix.

We use `f1_score` and confusion matrix to determine performance on the test batches of each dataset.

## File breakdown

### `main_rob.py`
Driver file, responsible for putting together the code that makes datasets and the code which trains and tests. The `make_datasets` function makes use of the auxiliary files in the folder `dataset_management` and creates datasets with images to which are applied standard transformations. For convenience, we don't also generate the tasks themselves here but we do this once independently and then only recall them. Next there is a part handling reinitalization of the model and the metrics arrays with previous values if that is wanted. The `train_datasets` function takes care of making the transiting between training consecutive tasks (the training of a task itself is taken care of by the `Trainer` class); for example it modifies the Fisher information matrix if that is required and resets the optimizer.

### `Trainer.py`
The `Trainer` class is capable to self-sufficiently train a single task and make evaluations on the test dataset once training is stopped. Mostly everything about training is a parameter of the class: the learning algorithm, learning rate and number of epochs, for example. The training and testing over single epochs are handled by the methods `train_epoch` and `test_epoch` respectively. The `train` method uses these to train and test over a single epoch. At the end of each epoch we then also test performance on all the previous tasks. 

### `dataset_management`
The `task_creation.py` file contains the code which suggests tasks and their components in terms of classes. The `consolidate_parents` function simplifies the semantic tree so that each node has only the node with the most number of children amongst its parents as its parent. To prepare the tree for DFS, it is saved as a class `DFSLeafCount` which contains additional information such as the marked vertices. DFS finding the tasks is done with the `dfs_superclass_finder` function. The `dataset_creation.py` file contains the `TinyImagenetTask` class which inherits the `torchvision.ImageFolder` one to handle datasets. The functions `get_cl_dset` and `get_tasks` are auxiliaries converting the tasks saved in a file to an array and getting a single task out of the array, respectively.
### `Models.py`

Contains handmade implementations of the standard models LeNet and DenseNet. For simplicity these are made of several instances of the class `Block` (resp. `Block2`); these "blocks" are the ones handling convolutions, all that sits between such blocks are maxpools.

### `plotting_utils.py` and `plotscript.py`
Strictly for plotting metrics. Employed separately from the driver code. The most relevant functions are `plot_acc_average_tasks` and `plot_train_rout_conf_entr` in `plotting_utils.py`. The first plots comparative graphs of `f1_score` evolution over the training of the several tasks and the second plots the evolution of confusion matrices (refer to the folder `plots` to see examples).
