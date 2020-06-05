import torch
import torch.utils.data as data
import csv
import os, math
import sys
import random
import imageio
import json
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import scipy.io
import glob
import matplotlib.pyplot as plt
import datasets
import platform
if platform.python_version().startswith('3'):
    import three
from fcn.config import cfg
from pathlib import Path

# ShapeNet uses +Y as up. YCB uses +Z as up. Swap these.
OBJ_DEFAULT_POSE = torch.tensor((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))

_package_dir = Path(os.path.dirname(os.path.realpath(__file__)))
_resources_dir = _package_dir / 'resources'

def load_roughness_values():
    path = _resources_dir / 'merl_blinn_phong.csv'
    with path.open('r') as f:
        reader = csv.reader(f)
        glossiness = np.array([float(row[-1]) for row in reader])

    roughness = (2 / (glossiness + 2)) ** (1.0 / 4.0)
    return roughness


def load_taxonomy():
    with (_resources_dir / 'shapenet_taxonomy.json').open('r') as f:
        taxonomy = json.load(f)

    taxonomy = {d['synsetId']: d for d in taxonomy}
    return taxonomy


def gather_synset_ids(taxonomy, synset_id):
    synset_ids = []
    stack = [synset_id]
    while len(stack) > 0:
        current_id = stack.pop()
        synset_ids.append(current_id)
        synset = taxonomy[current_id]
        stack.extend(synset['children'])

    return synset_ids


def category_to_synset_ids(taxonomy, category, include_children=True):
    synset_ids = []
    for synset_id, synset_dict in taxonomy.items():
        names = synset_dict['name'].split(',')
        if category in names:
            if include_children:
                synset_ids.extend(gather_synset_ids(taxonomy, synset_id))
            else:
                synset_ids.append(synset_id)

    return synset_ids


def get_shape_paths_categories(dataset_dir, categories=None):
    """
    Returns shape paths for ShapeNet.

    Args:
        dataset_dir: the directory containing the dataset
        blacklist_synsets: a list of synsets to exclude

    Returns:

    """
    shape_index_path = (dataset_dir / 'paths.txt')
    if shape_index_path.exists():
        with shape_index_path.open('r') as f:
            paths = [Path(dataset_dir, p.strip()) for p in f.readlines()]
    else:
        paths = list(dataset_dir.glob('**/uv_unwrapped.obj'))

    if categories is not None:

        taxonomy = load_taxonomy()
        synsets = set()
        for category in categories:
            synsets.update(category_to_synset_ids(taxonomy, category))
        print('selected synsets', synsets)

        num_selected = sum(1 for p in paths if p.parent.parent.parent.name in synsets)
        paths = [p for p in paths
                 if p.parent.parent.parent.name in synsets]
        print(categories)
        print("selected shapes %d" % (num_selected))

    return paths


def get_shape_paths(dataset_dir, blacklist_synsets=None):
    """
    Returns shape paths for ShapeNet.

    Args:
        dataset_dir: the directory containing the dataset
        blacklist_synsets: a list of synsets to exclude

    Returns:

    """
    shape_index_path = (dataset_dir / 'paths.txt')
    if shape_index_path.exists():
        with shape_index_path.open('r') as f:
            paths = [Path(dataset_dir, p.strip()) for p in f.readlines()]
    else:
        paths = list(dataset_dir.glob('**/uv_unwrapped.obj'))

    if blacklist_synsets is not None:
        num_filtered = sum(1 for p in paths if p.parent.parent.parent.name in blacklist_synsets)
        paths = [p for p in paths
                 if p.parent.parent.parent.name not in blacklist_synsets]
        print("filtered blacklisted shapes %d, remaining %d" % (num_filtered, len(paths)))

    return paths


def index_paths(dataset_dir, ext, index_name='paths.txt'):
    index_path = (dataset_dir / index_name)
    print(dataset_dir)
    if index_path.exists():
        with open(index_path, 'r') as f:
            return [Path(dataset_dir, p.strip()) for p in f.readlines()]
    else:
        return list(dataset_dir.glob('*' + ext))
