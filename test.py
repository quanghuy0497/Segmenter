import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP

import config
import utils.torch as ptu
from model.factory import load_model
from utils import distributed
from engine import eval_dataset


@click.command()
@click.option("--model_path", type=str)
@click.option("--dataset", type=str)
@click.option("--im-size", default=None, type=int)
@click.option("--multiscale/--singlescale", default=False, is_flag=True)
@click.option("--blend/--no-blend", default=True, is_flag=True)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--window-batch-size", default=4, type=int)
@click.option("--save-images", default=False, is_flag=True)
@click.option("--frac-dataset", default=1.0, type=float)
@click.option("--combine", default=False, is_flag=True)

def main(
    model_path,
    dataset,
    im_size,
    multiscale,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    save_images,
    combine,
    frac_dataset):
    
    model_path = "logs/" + model_path
    model_dir = Path(model_path)

    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.init_process()

    model, variant = load_model(model_path)
    patch_size = model.patch_size
    model.eval()
    model.to(ptu.device)
    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    cfg = config.load_config()
    dataset_cfg = cfg["dataset"][dataset]
    normalization = variant["dataset_kwargs"]["normalization"]
    if im_size is None:
        im_size = dataset_cfg.get("im_size", variant["dataset_kwargs"]["image_size"])
    if window_size is None:
        window_size = variant["dataset_kwargs"]["crop_size"]
    if window_stride is None:
        window_stride = variant["dataset_kwargs"]["crop_size"] - 32

    dataset_kwargs = dict(
        dataset=dataset,
        image_size=im_size,
        crop_size=im_size,
        patch_size=patch_size,
        batch_size=1,
        num_workers=8,
        split="test",
        normalization=normalization,
        crop=False,
        rep_aug=False)

    eval_dataset(
        model,
        multiscale,
        model_dir,
        blend,
        window_size,
        window_stride,
        window_batch_size,
        save_images,
        frac_dataset,
        dataset_kwargs,
        combine)

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    main()
