import os
import sys
import glob
import json
import datetime
import argparse
import traceback
import requests
from tqdm import tqdm
from typing import List, Optional, Union

import tator
import random
import numpy as np
import pandas as pd

import cv2
import torch
import torchvision.ops.boxes as bops

import mmcv
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.structures import TrackDataSample
from mmdet.structures import DetDataSample
from mmengine.structures import PixelData
from mmengine.structures import InstanceData
from mmyolo.registry import VISUALIZERS

from mmengine.utils import track_iter_progress

from boxmot import BYTETracker

import norfair
from norfair import Detection, Tracker, Video, get_cutout
from norfair.filter import OptimizedKalmanFilterFactory

from Toolbox.Tools.SAM import *

import warnings

warnings.filterwarnings("ignore")

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000


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


def get_color(index):
    """
    :param index:
    :return:
    """
    # Set the random seed to ensure consistent colors for the same index
    random.seed((42, index))

    # Generate random RGB values
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    return red, green, blue


def get_hist(image):
    """
    :param image:
    :return:
    """
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)],
        [0, 1],
        None,
        [128, 128],
        [0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()


def embedding_distance(matched_not_init_trackers, unmatched_trackers):
    """
    :param matched_not_init_trackers:
    :param unmatched_trackers:
    :return:
    """
    snd_embedding = unmatched_trackers.last_detection.embedding

    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1

    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue

        distance = 1 - cv2.compareHist(snd_embedding,
                                       detection_fst.embedding,
                                       cv2.HISTCMP_CORREL)
        if distance < 0.5:
            return distance

    return 1


def detections_to_norfair_detections(detections, class_map, threshold):
    """
    :param detections:
    :return:
    """
    bboxes = []
    norfair_detections = []

    for predictions in detections.pred_instances:

        # Parse out the predictions
        pred_box = predictions.bboxes[0]
        pred_score = predictions.scores.item()
        pred_label = predictions.labels.item()

        if pred_score < threshold:
            continue

        # Calculate bounding box
        bbox = np.array(
            [
                [pred_box[0].item(), pred_box[1].item()],
                [pred_box[2].item(), pred_box[3].item()],
            ]
        )
        scores = np.array(
            [pred_score, pred_score]
        )
        norfair_detections.append(
            Detection(
                points=bbox, scores=scores, label=class_map[pred_label]
            )
        )
        bboxes.append(bbox)

    return norfair_detections, bboxes


def algorithm(args):
    """
    :param args:
    :return:
    """
    print("\n###############################################")
    print("Algorithm")
    print("###############################################\n")

    try:
        # Setting the api given the token, authentication
        token = args.api_token
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        print(f"ERROR: Could not authenticate with provided API Token\n{e}")
        sys.exit(1)

    # Project id containing media
    if args.project_id:
        project_id = args.project_id
    else:
        print(f"ERROR: Project ID provided is invalid; please check input")
        sys.exit(1)

    # List of media
    if args.media_ids:
        media_ids = args.media_ids
    else:
        print(f"ERROR: Medias provided is invalid; please check input")
        sys.exit(1)

    # Check that config file exists
    if os.path.exists(args.run_dir):
        run_dir = f"{args.run_dir}\\"
        run_name = os.path.dirname(run_dir).split("\\")[-1]
        print(f"NOTE: Using run {run_name}")
    else:
        print(f"ERROR: Run directory doesn't exist; please check input")
        sys.exit(1)

    try:
        # Find the config file
        config = glob.glob(f"{run_dir}*.py")[0]
        config_name = os.path.basename(config).split(".")[0]
        print(f"NOTE: Using config file {config_name}")
    except:
        print(f"ERROR: Config file doesn't exist; please check input")
        sys.exit(1)

    try:
        # Find the checkpoint file
        checkpoints = glob.glob(f"{run_dir}*.pth")
        checkpoint = [c for c in checkpoints if os.path.basename(c) == f'epoch_{args.epoch}.pth'][0]
        checkpoint_name = os.path.basename(checkpoint).split(".")[0]
        print(f"NOTE: Using checkpoint file {checkpoint_name}")
    except:
        print(f"ERROR: Checkpoint file doesn't exist; please check input")
        sys.exit(1)

    try:
        # Find the class map json file
        class_map = f"{run_dir}class_map.json"

        with open(class_map, 'r') as input_file:
            class_map = json.load(input_file)

        # Update class map format
        class_map = {d['id']: d['name'] for d in class_map}
    except:
        print(f"ERROR: Class Map JSON file provided doesn't exist; please check input")
        sys.exit(1)

    try:
        # The type of localization for the project (bounding box, attributes)
        loc_type_id = 440  # Detection Box
        loc_name = api.get_localization_type(loc_type_id).name
        layer_type_id = 228  # AI Experiments
        layer_name = api.get_version(layer_type_id).name
        state_type_id = 438  # State Type
        state_name = api.get_state_type(state_type_id).name
    except Exception as e:
        print(f"ERROR: Could not find the correct localization type in project {project_id}")
        sys.exit(1)

    # Root where all output is saved
    output_dir = f"{args.output_dir}\\"

    # ------------------------------------------------
    # Model config
    # ------------------------------------------------
    print(f"NOTE: Setting up model...")

    # Either GPU or CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"NOTE: Using device {device}")

    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, palette='random', device=device)

    # build test pipeline
    model.cfg.test_cfg = dict(max_per_img=300,
                              min_bbox_size=0,
                              nms=dict(iou_threshold=args.nms_threshold, type='nms'),
                              nms_pre=30000,
                              score_thr=args.pred_threshold)

    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # ------------------------------------------------
    # Track
    # ------------------------------------------------
    if args.track:

        # Tracker object
        tracker = Tracker(distance_function="iou",
                          distance_threshold=DISTANCE_THRESHOLD_BBOX)

    # ------------------------------------------------
    # Media Loop
    # ------------------------------------------------
    for media_id in media_ids:

        # ------------------------------------------------
        # Download media
        # ------------------------------------------------
        try:
            # Get the video handler
            media = api.get_media(media_id)
            ext = media.name.split(".")[-1]
            media_dir = f"{output_dir}Algorithm_{get_now()}_{media_id}\\"
            video_path = f"{media_dir}{media_id}.{ext}"
            os.makedirs(media_dir, exist_ok=True)
            print(f"NOTE: Downloading {media.name}...")
        except Exception as e:
            print(f"ERROR: Could not get media from {media_id}")
            sys.exit(1)

        # Download the video
        for progress in tator.util.download_media(api, media, video_path):
            print(f"NOTE: Download progress: {progress}%")

        # Check that video was downloaded
        if os.path.exists(video_path):
            print(f"NOTE: Media {media.name} downloaded successfully")
        else:
            print(f"ERROR: Media {media.name} did not download successfully; skipping")
            continue

        # Do inference on each video.
        print(f"NOTE: Doing inference on {media.name}")

        # Create the video handler
        video = Video(input_path=video_path)

        # Loop through all the frames in the video
        for f_idx, frame in tqdm(enumerate(video)):

            # Make predictions on every N frames within window
            if f_idx % args.every_n == 0 and f_idx > args.start_at:

                # Make predictions
                detections = inference_detector(model, frame, test_pipeline=test_pipeline)
                # Convert detections for tracking
                detections, bboxes = detections_to_norfair_detections(detections, class_map, args.pred_threshold)

                # Update and get tracked objects
                tracked_objects = tracker.update(detections=detections,
                                                 period=args.every_n)

                # Draw objects on current frame
                norfair.draw_boxes(frame,
                                   tracked_objects,
                                   draw_labels=True,
                                   draw_ids=True,
                                   color='by_id')

                # Write
                video.write(frame)

                if args.show_video:
                    video.show(frame)

    print(f"NOTE: Completed inference on {len(media_ids)} medias.")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Algorithm")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int,
                        help="Project ID for desired media")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory containing the run")

    parser.add_argument("--epoch", type=int, required=True,
                        help="Epoch N checkpoint to use")

    parser.add_argument("--every_n", type=int, default=30,
                        help="Make predictions on every N frames")

    parser.add_argument("--start_at", type=int, default=0,
                        help="Frame to start making predictions at")

    parser.add_argument('--pred_threshold', type=float, default=0.3,
                        help='Prediction confidence threshold')

    parser.add_argument('--nms_threshold', type=float, default=0.35,
                        help='Non-maximum suppression threshold (low is conservative)')

    parser.add_argument('--track', action='store_true',
                        help='Track objects')

    parser.add_argument('--show_video', action='store_true',
                        help='Show video, and save it to the predictions directory')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.abspath("../../../Data/Predictions/"),
                        help='Directory to save output from model')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'

    try:
        algorithm(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
