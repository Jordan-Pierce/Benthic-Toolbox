import os
import sys
import argparse
import traceback

import json
import glob

import matplotlib
import requests

import numpy as np
import matplotlib.cm as cm

from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

# Where all checkpoints are stored
CHECKPOINT_DIR = f"./checkpoints/"


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def download_checkpoint(url, path):
    """
    :param url:
    :param path:
    :return:
    """

    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                file.write(response.content)
            print(f"NOTE: Downloaded to {path}")

        else:
            raise Exception

    # If it didn't download, exit.
    except Exception as e:
        print(f"ERROR: Failed to download file")
        sys.exit(1)


def get_metainfo(class_map):
    """
    :param class_map:
    :return:
    """
    # Get the class names
    class_names = [d['name'] for d in class_map]

    # Generate the colormap using matplotlib
    cmap = matplotlib.colormaps['tab20']
    pal = [cmap(i)[0:3] for i in range(len(class_names))]
    pal = [(np.array(p) * 255).astype(np.uint8).tolist() for p in pal]

    # Create metainfo
    metainfo = {'classes': tuple(class_names),
                'palette': pal}

    return metainfo


def train(args):
    """
    :param args:
    :return:
    """
    print("\n###############################################")
    print("Train")
    print("###############################################\n")

    # Reduce the number of repeated compilations
    # and improve training speed.
    setup_cache_size_limit_of_dynamo()

    # Set the variables
    if os.path.exists(args.annotations):
        annotations = os.path.basename(args.annotations)
    else:
        print(f"ERROR: Annotation file does not exist; check input provided")
        sys.exit(1)

    if os.path.exists(args.class_map):
        with open(args.class_map, 'r') as input_file:
            class_map = json.load(input_file)

        print(f"NOTE: Creating metainfo")
        metainfo = get_metainfo(class_map)

    else:
        print(f"ERROR: Annotation file does not exist; check input provided")
        sys.exit(1)

    if os.path.exists(args.config):
        config_file = args.config
    else:
        print(f"ERROR: Configuration file does not exist; check input provided")
        sys.exit(1)

    print(f"NOTE: Using config file {os.path.basename(config_file)}")
    # Create the config
    cfg = Config.fromfile(config_file)

    # Create the output folder
    output_dir = args.output_dir
    work_dir = f"{output_dir}\\model\\"
    os.makedirs(work_dir, exist_ok=True)
    cfg.work_dir = work_dir

    # Check that the expected weights are there
    checkpoint_url = cfg.checkpoint
    checkpoint_path = f"{CHECKPOINT_DIR}{os.path.basename(checkpoint_url)}"

    # If checkpoint does not exist, download it
    if not os.path.exists(checkpoint_path):
        print(f"NOTE: Downloading {os.path.basename(checkpoint_url)}")
        download_checkpoint(checkpoint_url, checkpoint_path)

    print(f"NOTE: Loading checkpoint {os.path.basename(checkpoint_path)}")
    # Set the path of the checkpoint
    cfg.load_from = checkpoint_path

    # Training parameters
    train_batch_size_per_gpu = 16
    max_epochs = 500
    checkpoint_interval = 100
    base_lr = 0.00008

    print(f"NOTE: Setting training parameters")
    cfg.max_epochs = max_epochs
    cfg.default_hooks['checkpoint']['interval'] = checkpoint_interval

    print(f"NOTE: Setting data root to {args.data_root}")
    # Path to dataset
    cfg.data_root = args.data_root

    print(f"NOTE: Number of class categories {len(class_map)}")
    # Number of classes
    cfg.model['bbox_head']['num_classes'] = len(class_map)

    print(f"NOTE: Creating train dataloader")
    # Data Loaders
    cfg.train_dataloader['batch_size'] = train_batch_size_per_gpu
    cfg.train_dataloader['dataset']['data_root'] = f"{args.data_root}\\"
    cfg.train_dataloader['dataset']['ann_file'] = annotations
    cfg.train_dataloader['dataset']['data_prefix'] = {'img': 'frames\\'}
    cfg.train_dataloader['dataset']['metainfo'] = metainfo

    print(f"NOTE: Creating valid dataloader")
    # Valid dataloader
    cfg.val_dataloader['batch_size'] = 1
    cfg.val_dataloader['dataset']['data_root'] = f"{args.data_root}\\"
    cfg.val_dataloader['dataset']['ann_file'] = annotations
    cfg.val_dataloader['dataset']['data_prefix'] = {'img': 'frames\\'}
    cfg.val_dataloader['dataset']['metainfo'] = metainfo

    print(f"NOTE: Creating test dataloader")
    # Test dataloader
    cfg.test_dataloader = cfg.val_dataloader

    print(f"NOTE: Creating evaluators")
    # Evaluators, make them the same
    cfg.val_evaluator['ann_file'] = f"{args.data_root}\\{annotations}"
    cfg.test_evaluator = cfg.val_evaluator

    print("NOTE: Setting launcher")
    # Launcher
    cfg.launcher = args.launcher

    print("NOTE: Setting autoscaler")
    # Enable automatically scaling LR
    if 'auto_scale_lr' in cfg and 'enable' in cfg.auto_scale_lr and 'base_batch_size' in cfg.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    print("NOTE: Building Runner")
    # Build the runner from config
    if 'runner_type' not in cfg:
        # Build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # Build customized runner from the registry
        runner = RUNNERS.build(cfg)

    try:
        print("NOTE: Starting training")
        # start training
        runner.train()
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument("--config", type=str,
                        default="./configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
                        help="Path to model config file")

    parser.add_argument("--annotations", type=str,
                        help="Path to the COCO formatted annotations")

    parser.add_argument("--class_map", type=str,
                        help="Path to the Class Map JSON file")

    parser.add_argument('--data_root', type=str,
                        help='Directory where all data is saved')

    parser.add_argument('--output_dir', type=str,
                        help='Directory to save logs and models')

    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'

    try:
        train(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()

print(".")
