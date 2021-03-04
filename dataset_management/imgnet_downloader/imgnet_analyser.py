import os
import numpy as np
import requests
import argparse
import json
import time
import logging
import csv

from multiprocessing import Pool, Process, Value, Lock

from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL

basepath = "."
def get_hierarchy(basepath):
    """ Each line in `is_a.txt` is a parent -> child relation.
        Parsed 75850 rows.
        parents:   75850 ->  16693 unique.
        children:  75850 ->  74389 unique.
    """
    p2c = defaultdict(list)
    c2p = defaultdict(list)
    with open(basepath+"/data/imagenet/is_a.txt") as csv_file:
        for row_cnt, row in enumerate(csv.reader(csv_file, delimiter=" ")):
            #row is a (0,2)-vector; pos 0: parent ; pos 1: child
            p2c[row[0]].append(row[1])
            c2p[row[1]].append(row[0])
    print(f"Parsed {row_cnt + 1} lines from imagenet `is_a.txt`.")
    print(f"p2c: {len(p2c)} parents.")
    print(f"c2p: {len(c2p)} children.")
    #getting only the leaves
    all_parents = set(p2c.keys())
    all_children = set(c2p.keys())
    leaves = all_children - all_parents
    print(f"leaves: {len(leaves)}")
    return p2c, c2p, leaves

def consolidate_parents(p2c, c2p):
    """
    For each child, keeps as its parent only the one with the largest number of children.
    Keeps as parents only the ones that are the parent with the largest number of children for at least one child
    """
    c2p_ = {}
    p2c_ = defaultdict(list)
    for child, parents in c2p.items():
        parent_, children_no = None, 0
        for parent in parents:
            if len(p2c[parent]) > children_no:
                parent_ = parent
                children_no = len(p2c[parent]) #ADDED BY VLAD
        c2p_[child] = parent_
        p2c_[parent_].append(child)
    print(f"consolidated p2c (only parents with maximal # leaves kept): {len(p2c_)} parents.")
    print(f"consolidated c2p (only parents with maximal # leaves kept): {len(c2p_)} children.")
    return p2c_, c2p_

def get_all_parents(wnids, c2p, wnid2words=None):
    """
    Looks at each element in wnids
    Takes all ancestors of each element in wnid (parents and parents of parents) and puts them in a list over the element in wnid
    Obs: In c2p we kept only ONE parent (the one with highest number of children).
    """
    all_parents = defaultdict(list)
    for cnt, wnid in enumerate(wnids):
        if wnid2words is not None:
            print(cnt, wnid, wnid2words[wnid])
        parent = c2p[wnid]
        while parent in c2p:
            all_parents[wnid].append(parent)
            parent = c2p[parent]
    return all_parents