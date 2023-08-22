import os
import sys
import glob
import json
import argparse
import traceback

import cv2
import torch
import tator
import numpy as np
import pandas as pd

import mmcv
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from mmdet.registry import VISUALIZERS
from mmengine.utils import track_iter_progress

from deep_sort_realtime.deepsort_tracker import DeepSort

import warnings

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def tracker(args):
    """

    :param args:
    :return:
    """
    print("\n###############################################")
    print("Tracker")
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
        layer_type_id = 228  # AI Experiments
        state_type_id = 147  # State Type
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

    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # ------------------------------------------------
    # Tracker
    # ------------------------------------------------
    # DeepSort object
    deepsort = DeepSort(max_age=50,
                        max_iou_distance=0.7,
                        max_cosine_distance=0.2,
                        n_init=3,
                        nn_budget=100,
                        nms_max_overlap=1.0,
                        embedder="clip_ViT-B/32",
                        embedder_gpu=device)

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # Loop through medias
    for media_id in media_ids:

        # ------------------------------------------------
        # Download media
        # ------------------------------------------------
        # For upload
        localizations = []
        # For local archive
        predictions = []

        try:
            # Get the video handler
            media = api.get_media(media_id)
            ext = media.name.split(".")[-1]
            output_media_dir = f"{output_dir}Detector_{media_id}\\"
            output_video_path = f"{output_media_dir}{media_id}.{ext}"
            os.makedirs(output_media_dir, exist_ok=True)
            print(f"NOTE: Downloading {media.name}...")
        except Exception as e:
            print(f"ERROR: Could not get media from {media_id}")
            sys.exit(1)

        # Download the video
        for progress in tator.util.download_media(api, media, output_video_path):
            print(f"NOTE: Download progress: {progress}%")

        # Check that video was downloaded
        if os.path.exists(output_video_path):
            print(f"NOTE: Media {media.name} downloaded successfully")
        else:
            print(f"ERROR: Media {media.name} did not download successfully; skipping")
            continue

        # Do inference on each video.
        print(f"NOTE: Doing inference on {media.name}")

        # Create the video handler
        video_reader = mmcv.VideoReader(output_video_path)
        video_writer = None

        if args.show_video:
            # Create a path for the predictions video
            output_pred_path = output_video_path.split(".")[0] + "_tracking.mp4"
            # To output the video with predictions super-imposed
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_pred_path, fourcc, video_reader.fps,
                                           (video_reader.width, video_reader.height))

        # Loop through all the frames in the video
        for f_idx, frame in enumerate(track_iter_progress(video_reader)):

            # Make predictions on every N frames
            if f_idx % args.every_n == 0:

                # Make predictions
                result = inference_detector(model, frame, test_pipeline=test_pipeline)

                # Parse out predictions
                scores = result.pred_instances.cpu().detach().scores.numpy()
                labels = result.pred_instances.cpu().detach().labels.numpy()
                bboxes = result.pred_instances.cpu().detach().bboxes.numpy()

                # Filter based on threshold
                indices = np.where(scores >= args.pred_threshold)[0]
                scores = scores[indices]
                labels = labels[indices]
                bboxes = bboxes[indices]

                # Reformat result
                detections = []

                for i in range(len(indices)):
                    detections.append([bboxes[i], scores[i], labels[i]])

                # Update tracker
                tracks = deepsort.update_tracks(detections, frame=frame)

                # Update result
                updated_scores = []
                updated_bboxes = []
                updated_labels = []
                track_ids = []

                # Loop through tracks
                for t_idx, track in enumerate(tracks):

                    if not track.get_det_conf():
                        updated_scores.append(args.pred_threshold)
                        updated_bboxes.append(track.to_ltwh())
                        updated_labels.append(track.get_det_class())
                        track_ids.append(int(track.track_id))
                    else:
                        updated_scores.append(track.get_det_conf())
                        updated_bboxes.append(track.to_ltwh(orig=True))
                        updated_labels.append(track.get_det_class())
                        track_ids.append(int(track.track_id))

                # Create a new data sample after filtering (for local visualization only)
                pred_track_instances = InstanceData(metainfo=result.metainfo)
                pred_track_instances.scores = np.array(updated_scores)
                pred_track_instances.bboxes = np.array(updated_bboxes)
                pred_track_instances.labels = np.array(track_ids)
                result = DetDataSample(pred_instances=pred_track_instances)

                # Record the predictions in tator format
                for i_idx in range(len(updated_bboxes)):
                    # Score, Label for tator
                    score = updated_scores[i_idx]
                    label = class_map[updated_labels[i_idx]]

                    # Redundant sanity check
                    if score <= args.pred_threshold:
                        continue

                    # Local archive format
                    xmin = int(updated_bboxes[i_idx][0])
                    ymin = int(updated_bboxes[i_idx][1])
                    xmax = int(updated_bboxes[i_idx][2])
                    ymax = int(updated_bboxes[i_idx][3])

                    # Tator format of bounding boxes
                    x = float(xmin / video_reader.width)
                    y = float(ymin / video_reader.height)
                    w = float((xmax - xmin) / video_reader.width)
                    h = float((ymax - ymin) / video_reader.height)

                    # For tator upload
                    loc = {'media_id': media.id,
                           'type': loc_type_id,
                           'version': layer_type_id,
                           'x': x,
                           'y': y,
                           'width': w,
                           'height': h,
                           'frame': f_idx,
                           'track_id': track_ids[i_idx],
                           'attributes': {
                               'ScientificName': "",
                               'CommonName': "",
                               'Notes': "",
                               'Needs Review': True,
                               'Score': score}
                           }

                    # For local archive
                    pred = [f_idx, label, score, xmin, ymin, xmax, ymax]

                    # Add to lists
                    localizations.append(loc)
                    predictions.append(pred)

                # Showing video
                if args.show_video:

                    try:
                        # This is just used for visualizations locally
                        class_tracker = {}

                        # Update the classes for tracking objects
                        for track in deepsort.tracker.tracks:
                            class_tracker[f'Object {str(track.track_id)}'] = track.track_id

                        visualizer.dataset_meta['classes'] = list(class_tracker.keys())

                        # Add predictions to frame
                        visualizer.add_datasample(
                            name='video',
                            image=frame,
                            data_sample=result,
                            draw_gt=False,
                            show=False,
                            wait_time=0.1,
                            pred_score_thr=args.pred_threshold)
                    except:
                        pass

                    # Get the updated frame with visuals
                    frame = visualizer.get_image()

                    # Display predictions as they are happening
                    cv2.namedWindow('video', 0)
                    mmcv.imshow(frame, 'video', 1)

                    # Write the frame to video file
                    video_writer.write(frame)

        if args.show_video:
            # Release the video handler
            video_writer.release()
            video_writer = None

            # Close the viewing window
            cv2.destroyAllWindows()

        try:
            # Close the video writer
            video_reader = None
            # Delete the original video
            os.remove(output_video_path)
        except Exception as e:
            print(f"WARNING: Could not delete media\n{e}")

        # ------------------------------------------------
        # Save and Upload localizations for this media
        # ------------------------------------------------

        # Save merged annotations as "predictions.csv"
        predictions_path = f"{output_media_dir}predictions.csv"
        predictions = pd.DataFrame(predictions, columns=['Frame', 'Label', 'Score', 'xmin', 'ymin', 'xmax', 'ymax'])
        predictions.to_csv(predictions_path)
        print(f"NOTE: Merged predictions saved to {os.path.basename(predictions_path)}")

        if args.upload:
            # Create the localizations in the video.
            print(f"NOTE: Uploading {len(localizations)} detections on {media.name}...")
            localization_ids = []
            for response in tator.util.chunked_create(api.create_localization_list, project_id, body=localizations):
                localization_ids.extend(response.id)
            print(f"NOTE: Successfully created {len(localization_ids)} localizations on {media.name}!")

            # Associate the localization ids with track ids
            track_ids = [l['track_id'] for l in localizations]
            track_ids = np.array(list(zip(track_ids, localization_ids)))
            num_tracks = len(np.unique(track_ids.T[0]))

            print(f"NOTE: Uploading {num_tracks} tracks on {media.name}...")
            states = []
            for track_id in np.unique(track_ids.T[0]):

                state = {'type': state_type_id,
                         'version': layer_type_id,
                         'localization_ids': track_ids.T[1][np.where(track_ids.T[0] == track_id)].tolist(),
                         'media_ids': [media_id],
                         'Scientfic Name': "",
                         'Notes': ""}

                states.append(state)

            state_ids = []
            for response in tator.util.chunked_create(api.create_state_list, project_id, body=states):
                state_ids += response.id
            print(f"NOTE: Successfully created {len(state_ids)} tracks on {media.name}!")

    print(f"NOTE: Completed inference on {len(media_ids)} medias.")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Detector")

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
                        help="Epoch checkpoint to use")

    parser.add_argument("--every_n", type=int, default=30,
                        help="Make predictions on every N frames")

    parser.add_argument('--pred_threshold', type=float, default=0.3,
                        help='Prediction confidence threshold')

    parser.add_argument('--nms_threshold', type=float, default=0.65,
                        help='Non-maximum suppression threshold (low is conservative')

    parser.add_argument('--show_video', action='store_true',
                        help='Show video, and save it to the predictions directory')

    parser.add_argument('--upload', action='store_true',
                        help='Upload predictions to tator')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.abspath("../../../Data/Predictions/"),
                        help='Directory to save output from model')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'

    try:
        tracker(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
