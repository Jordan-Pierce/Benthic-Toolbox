import os
import sys
import json
import shutil
import datetime
import argparse
import traceback

import matplotlib
import numpy as np

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config
from mmengine.runner import Runner

# MMDet
from mmengine.registry import RUNNERS as MMDET_RUNNERS

# MMYolo
from mmyolo.registry import RUNNERS as MMYOLO_RUNNERS


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def get_now():
    """
    :return:
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    return now


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

    # Set the annotation files
    if os.path.exists(args.train) and os.path.exists(args.valid) and os.path.exists(args.test):
        train_annotations = args.train
        valid_annotations = args.valid
        test_annotations = args.test
    else:
        print(f"ERROR: The annotation files do no not exist; check input provided")
        sys.exit(1)

    # Set the class map, and then create the metainfo
    if os.path.exists(args.class_map):
        with open(args.class_map, 'r') as input_file:
            class_map = json.load(input_file)

        print(f"NOTE: Creating metainfo")
        metainfo = get_metainfo(class_map)

    else:
        print(f"ERROR: Annotation file does not exist; check input provided")
        sys.exit(1)

    # Set the class weights
    if os.path.exists(args.class_weights):
        with open(args.class_weights, 'r') as input_file:
            weights = json.load(input_file)

        print(f"NOTE: Creating class weights")
        class_weights = [weights[_['name']] for _ in class_map]

    else:
        print(f"ERROR: Annotation file does not exist; check input provided")
        sys.exit(1)

    # Set the config file
    if os.path.exists(args.config):
        config_file = args.config
        config_name = os.path.basename(config_file).split(".")[0]
    else:
        print(f"ERROR: Configuration file does not exist; check input provided")
        sys.exit(1)

    print(f"NOTE: Using config file {config_name}")
    # Create the config
    cfg = Config.fromfile(config_file)

    # Optional run name
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{get_now()}_{config_name}"

    # Create the output folder
    output_dir = f"{args.output_dir}\\"
    # Directory for run output
    work_dir = f"{output_dir}{run_name}\\"
    os.makedirs(work_dir, exist_ok=True)
    cfg.work_dir = work_dir

    # Save dataset json files for posterity
    shutil.copy(args.class_map, f'{work_dir}{os.path.basename(args.class_map)}')
    shutil.copy(args.train, f'{work_dir}{os.path.basename(args.train)}')
    shutil.copy(args.valid, f'{work_dir}{os.path.basename(args.valid)}')
    shutil.copy(args.test, f'{work_dir}{os.path.basename(args.test)}')

    # Training parameters
    base_lr = args.lr
    val_interval = 30
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    loss_weight = class_weights

    print(f"NOTE: Setting training parameters")
    cfg.max_epochs = max_epochs
    cfg.default_hooks['checkpoint']['interval'] = val_interval
    cfg.default_hooks['checkpoint']['max_keep_ckpts'] = val_interval
    cfg.optim_wrapper['optimizer']['lr'] = base_lr

    print(f"NOTE: Setting annotation files")
    # Path to dataset
    cfg.data_root = ""

    print(f"NOTE: Number of class categories {len(class_map)}")
    # Number of classes (MMYolo, or MMDet)
    if 'head_module' in cfg.model['bbox_head']:
        cfg.model['bbox_head']['head_module']['num_classes'] = len(class_map)
        cfg.model['train_cfg']['assigner']['num_classes'] = len(class_map)
    else:
        cfg.model['bbox_head']['num_classes'] = len(class_map)

    # Class weights
    cfg.model['train_cfg']['pos_weight'] = class_weights

    print(f"NOTE: Creating train dataloader")
    cfg.train_cfg['max_epochs'] = max_epochs
    cfg.train_cfg['val_interval'] = val_interval
    cfg.train_dataloader['batch_size'] = batch_size
    cfg.train_dataloader['dataset']['data_root'] = ""
    cfg.train_dataloader['dataset']['ann_file'] = train_annotations
    cfg.train_dataloader['dataset']['data_prefix'] = {'img': ''}
    cfg.train_dataloader['dataset']['metainfo'] = metainfo

    print(f"NOTE: Creating valid dataloader")
    cfg.val_dataloader['batch_size'] = batch_size
    cfg.val_dataloader['dataset']['data_root'] = ""
    cfg.val_dataloader['dataset']['ann_file'] = valid_annotations
    cfg.val_dataloader['dataset']['data_prefix'] = {'img': ''}
    cfg.val_dataloader['dataset']['metainfo'] = metainfo

    print(f"NOTE: Creating test dataloader")
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_dataloader['dataset']['ann_file'] = test_annotations

    print(f"NOTE: Creating evaluators")
    # Evaluators, make them the same
    cfg.val_evaluator['ann_file'] = valid_annotations
    cfg.val_evaluator['classwise'] = True
    cfg.test_evaluator['ann_file'] = test_annotations
    cfg.test_evaluator['classwise'] = True

    print(f"NOTE: Setting up Tensorboard ")

    cfg.visualizer['vis_backends'].append({'type': 'TensorboardVisBackend'})

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
        if 'yolo' in config_name.lower():
            runner = MMYOLO_RUNNERS.build(cfg)
        else:
            runner = MMDET_RUNNERS.build(cfg)

    try:
        print("NOTE: Starting training")
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
                        default="./configs/mmyolo/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py",
                        help="Path to model config file")

    parser.add_argument("--train", type=str, required=True,
                        help="Path to the COCO formatted annotations")

    parser.add_argument("--valid", type=str, required=True,
                        help="Path to the COCO formatted annotations")

    parser.add_argument("--test", type=str, required=True,
                        help="Path to the COCO formatted annotations")

    parser.add_argument("--class_map", type=str, required=True,
                        help="Path to the Class Map JSON file")

    parser.add_argument("--class_weights", type=str, required=False,
                        help="Path to the Class Weights file")

    parser.add_argument("--run_name", type=str, default="",
                        help="Name for run (optional)")

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save logs and models')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of samples to pass model in a single batch (GPU dependent)')

    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Total number of times model sees every sample in training set')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='The amount to adjust model parameters by during back-prop')

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
