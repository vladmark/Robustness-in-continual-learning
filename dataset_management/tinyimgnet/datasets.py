from torchvision.datasets import ImageFolder
import os
import csv
###Getting proper classes for tasks

def get_cl_dset(fp):
    """
    returns: dictionary: superclass and
    """
    cl_dset = {}
    with open(fp, "r") as f:
        cls_no, misc_cls, task_no = list(map(int, f.readline().split(",")))
        #first line is of the form: number of tasks, number of misc, number of min (?) classes per task
        for line in f.readlines():
            wnids = [wnid.strip() for wnid in line.split(",")]
            supercls = wnids.pop(0) #first id is actually a superclass???? makes no sense (classes in a task are suppposed to be from disjoint superclasses)
            cl_dset[supercls] = wnids
    cl_dset["meta"] = {"cls_no": cls_no, "task_no": task_no, "misc": misc_cls}
    print(f"\we have {len(cl_dset.keys())} tasks")
    return cl_dset


def get_task(cl_dset, task_id, basepath, verbose=True):
    """
        cl_dset: a
        dictionary
        containing
        -> key
        'meta': [number of classes / task, number of tasks, number of misc classes]
        -> other
        keys: the
        disjoint
        superclasses
        used in making
        of
        tasks(their
        number is the
        number
        of
        classes / task)
    """
    #construct dictionary: folder(class) {id: corresponding word}
    wnid2words = {
        r[0]: r[1]
        for r in csv.reader(
            open(basepath+"/data/tiny-imagenet-200/words.txt", "r"), delimiter="\t"
        )
    }

    #each
    task = [v[task_id] for k, v in cl_dset.items() if k != "meta"]
    if verbose:
      print("Superclasses used in tasks:")
      for node in cl_dset.keys():
        if node != 'meta':
          print(f"{node} -> {wnid2words[node]}")
      for i, cls_name in enumerate(task):
          print(i, cls_name, " ->  ", wnid2words[cls_name])
    return task

"""  ###Dataset class"""


class TinyImagenetTask(ImageFolder):
    def __init__(self, root, subset, **kwargs):
        self._subset = subset
        super(TinyImagenetTask, self).__init__(root, **kwargs)

    def _find_classes(self, dir):
        """ Finds the class folders in a dataset and filters out classes not
        appearing in the `subset`.

        Args:
            dir (string): Root directory path.
            subset (list): List of folders that make this particular dataset.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to
                (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        all_classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        for cls_name in self._subset:
            assert cls_name in all_classes, f"{cls_name} not in the root path."

        # `subset` ordering is the single source of truth
        # class idxs need to be consistent across subsets
        classes = [d for d in self._subset]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for k, v in class_to_idx.items():
            print(k, v)
        return classes, class_to_idx


