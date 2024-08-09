import argparse
import os

from get_architecture import get_architecture
from get_dataset import get_dataset


def get_args():
    parser = argparse.ArgumentParser(description="create a visualization of a model")

    model_selection = parser.add_argument_group("model selection")
    model_selection.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="name of the checkpoint directory",
    )

    args = parser.parse_args()
    return validate_args(args)


def validate_args(args):
    model_raw = args.model
    model_path = os.path.join(
        "/groups/funke/home/thambiduraiy/checkpoints/blinx/",
        model_raw,
        #         f"/groups/funke/home/{os.getlogin()}/checkpoints/blinx/", model_raw
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"couldn't find {model_path}")

    ds_name, ds_path, normalization_ds_path = get_dataset(model_raw)
    architecture = get_architecture(model_raw)

    return {
        "model_path": model_path,
        "model_name": model_raw,
        "ds_path": ds_path,
        "normalization_ds_path": normalization_ds_path,
        "architecture": architecture,
        "raw_args": args,
        "ds_name": ds_name,
    }
