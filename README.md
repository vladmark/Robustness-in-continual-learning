# Robust Features for Continual Learning

## Overview
In this project, we examine the possibility that semantic segmentation of an image database like ImageNet might mitigate the phenomenon of catastropic forgetting on simple successive classification tasks.

The goal is to learn a series of several image classification tasks (in our experiments, either 5 or 7 tasks) successively, where training on task $k$ is stopped once overfitting starts or good performance is reached. Once learning of task $k+1$ begins, there is no revisiting of images from tasks $1, \ldots, k$. The goal is to maintain good performance on tasks $1, \ldots, k$ while learning task $k+1$.

For each experiment, the number of classes to be learned supervisedly is fixed and is the same number $n_c$ for all tasks. Say we have $n_t$ tasks. The idea of semantic segmentation is that for each $i$ from $1$ to $n_c$, the classes $\{ c_i^1, \ldots, \ldots, c_i^{n_t} \}$ are semantically correlated, not just randomly chosen. For example, if in task 1 we learn "dog" as the first class, we might learn "cat" as the first class in task 2. Thus, the experiment becomes as much about catastrophic forgetting as about the amount of semantic information that is present in the images themselves.


## Algorithm description
Firstly, we take care of the problem of finding collections of tasks that are large enough in both the number of tasks and in the number of classes per task. For this, we use the semantic graph of ImageNet, where we keep for each node only its parent with the most number of children. Then we do a depth-first search to find the best candidates for "superclasses" i.e. collections of semantically correlated classes that can be used across different tasks, given several sets of constraints on both the number of classes per tasks and the number of tasks.

## Summary of files
