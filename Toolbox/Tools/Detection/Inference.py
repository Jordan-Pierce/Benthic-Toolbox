import os
import sys
import glob
import argparse
import traceback

import torch
from mmdet.apis import DetInferencer


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def inference(args):
    """

    :param args:
    :return:
    """
    # Check that config file exists
    if os.path.exists(args.config):
        config = args.config
        print(f"NOTE: Using config file {os.path.basename(config)}")
    else:
        print(f"ERROR: Config file doesn't exist; please check input")
        sys.exit(1)

    # Check that checkpoint is there
    if os.path.exists(args.checkpoint):
        checkpoint = args.checkpoint
        print(f"NOTE: Using checkpoint file {os.path.basename(checkpoint)}")
    else:
        print(f"ERROR: Checkpoint file doesn't exist; please check input")
        sys.exit(1)

    # Directory where media is stored
    if os.path.exists(args.media_dir):
        media_dir = args.media_dir
        media_name = os.path.basename(media_dir)
    else:
        print(f"ERROR: Media directory provided doesn't exist; please check input")
        sys.exit(1)

    # Collect the images from frames directory
    if os.path.exists(f"{media_dir}\\frames\\"):
        frame_dir = f"{media_dir}\\frames\\"
        frame_paths = glob.glob(f"{frame_dir}*.*")
        print(f"NOTE: Found {len(frame_paths)} frames in {media_name} directory")
    else:
        print(f"ERROR: Could not locate 'frames' directory in {media_dir}")
        sys.exit(1)

    # Output directory
    output_dir = f"{args.output_dir}\\{media_name}\\"
    os.makedirs(output_dir, exist_ok=True)

    # Either GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"NOTE: Using device {device}")

    try:
        # Load inference object
        inferencer = DetInferencer(model=config,
                                   weights=checkpoint,
                                   palette='coco',
                                   show_progress=True)

        print(f"NOTE: Weights {os.path.basename(checkpoint)} loaded successfully")

    except Exception as e:
        print(f"ERROR: Could not load model\n{e}")
        sys.exit(1)

    print(f"NOTE: Making predictors for {media_name}")

    try:
        # Dict of predictions, and visualization
        results = inferencer(frame_paths,
                             out_dir=output_dir,
                             no_save_pred=False,
                             pred_score_thr=args.pred_threshold)

    except Exception as e:
        print(f"ERROR: Could not make prediction on {media_name}\n{e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config file (.py)")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint file (.pth)")

    parser.add_argument('--media_dir', type=str, required=True,
                        help='Directory where all the data is located')

    parser.add_argument('--pred_threshold', type=float, default=0.25,
                        help='Prediction confidence threshold')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.abspath("../../../Data/Predictions/"),
                        help='Directory to save output from model')

    args = parser.parse_args()

    try:
        inference(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
