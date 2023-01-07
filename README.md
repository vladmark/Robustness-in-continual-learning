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

## Summary of files
