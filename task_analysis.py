from dataset_management.dataset_creation import get_cl_dset
from dataset_management.dataset_creation import get_task
import os

dataset_name = "imagenet"
datapath = os.path.join("data", dataset_name)
split_file_name = "cl_t5_c26"
cl_dset = get_cl_dset(os.path.join(datapath, split_file_name + ".txt"))
premade_tasks = None
img_counts = []
for task_no in range(cl_dset['meta']['task_no']):
    task = get_task(cl_dset, task_no, datapath, verbose=True) if premade_tasks is None else premade_tasks[
        task_no]
    task_img_counts = [len([img for img in os.listdir(os.path.join(datapath, "train", f"{cls_id}"))]) for cls_id in
                       task]
    img_counts.append(task_img_counts)
for task_no in range(cl_dset['meta']['task_no']):
    task_img_counts = img_counts[task_no]
    print(
        f"Task {task_no} has {sum(task_img_counts)} images in total, proportioned: {[(cls_cnt, '{:.2%}'.format(cls_cnt / float(sum(task_img_counts)))) for cls_cnt in task_img_counts]} .\n")
print(f"Dataset has on average (over all tasks and classes) {sum([sum(task_counts) for task_counts in img_counts])/sum([len(task_counts) for task_counts in img_counts])} images per class.")