import math
import shutil
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

import config
import utils.torch as ptu
import wandb as WandB
from data.factory import create_dataset
from data.utils import IGNORE_LABEL, rgb_denormalize, seg_to_rgb
from metrics import compute_metrics, gather_data
from model import utils
from model.factory import load_model
from model.utils import inference
from utils import distributed

def blend_im(im, seg, alpha=0.5):
    pil_im = Image.fromarray(im)
    pil_seg = Image.fromarray(seg)
    im_blend = Image.blend(pil_im, pil_seg, alpha).convert("RGB")
    return np.asarray(im_blend)


def save_im(save_dir, save_name, im, seg_pred, seg_gt, colors, blend, normalization):
    seg_rgb = seg_to_rgb(seg_gt[None], colors)
    pred_rgb = seg_to_rgb(seg_pred[None], colors)
    im_unnorm = rgb_denormalize(im, normalization)
    save_dir = Path(save_dir)

    # save images
    im_uint = (im_unnorm.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
    seg_rgb_uint = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
    seg_pred_uint = (255 * pred_rgb.cpu().numpy()).astype(np.uint8)
    for i in range(pred_rgb.shape[0]):
        if blend:
            blend_pred = blend_im(im_uint[i], seg_pred_uint[i])
            blend_gt = blend_im(im_uint[i], seg_rgb_uint[i])
            true_img, pred_img, gt_img = im_uint[i], blend_pred, blend_gt
        else:
            true_img, pred_img, gt_img = im_uint[i], seg_pred_uint[i], seg_rgb_uint[i]
        
        im_dir = save_dir / "result"
        im_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize = (80, 18))
        
        plt.subplot(131), plt.imshow(gt_img)
        plt.title('Ground Truth', fontsize = 60, pad = 30), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(true_img)
        plt.title('Original Image', fontsize = 60, pad = 30), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(pred_img)
        plt.title('Prediction', fontsize = 60, pad = 30), plt.xticks([]), plt.yticks([])
        plt.tight_layout()

        plt.savefig(str(im_dir) + "/" + save_name)
        plt.close()


def process_batch(model, batch, window_size, window_stride, window_batch_size):
    ims = batch["im"]
    ims_metas = batch["im_metas"]
    ori_shape = ims_metas[0]["ori_shape"]
    ori_shape = (ori_shape[0].item(), ori_shape[1].item())
    filename = batch["im_metas"][0]["ori_filename"][0]

    model_without_ddp = model
    if ptu.distributed:
        model_without_ddp = model.module
    seg_pred = inference(
        model_without_ddp,
        ims,
        ims_metas,
        ori_shape,
        window_size,
        window_stride,
        window_batch_size,
    )
    seg_pred = seg_pred.argmax(0)
    im = F.interpolate(ims[-1], ori_shape, mode="bilinear")

    return filename, im.cpu(), seg_pred.cpu()


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    num_epochs,
    wandb
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = {}
    avg_loss = 0
    avg_acc = 0
    
    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    
    with tqdm.tqdm(total = len(data_loader)) as train_pbar:
        for i, batch in enumerate(data_loader):  
            im = batch["im"].to(ptu.device)
            seg_gt = batch["segmentation"].long().to(ptu.device)

            with amp_autocast():
                seg_pred = model.forward(im)
                loss = criterion(seg_pred, seg_gt)
                avg_loss += loss.item()
                predict = torch.argmax(seg_pred, 1)
                acc = torch.sum(predict == seg_gt) / torch.Tensor.nelement(seg_gt)
                avg_acc += acc

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value), force=True)

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                )
            else:
                loss.backward()
                optimizer.step()

            num_updates += 1
            lr_scheduler.step_update(num_updates=num_updates)

            torch.cuda.synchronize()
            train_pbar.set_description('Epoch {:03d}/{:03d}: Pixel Acc {:.2f}  | Loss {:.6f}'.format(epoch + 1, num_epochs, avg_acc/float(i + 1) * 100, avg_loss/float(i+1)))
            train_pbar.update(1)
            
    if wandb:
        WandB.log({"Train/Loss": avg_loss/len(data_loader),'Train/Pixel Acc': avg_acc/len(data_loader) * 100},  step=epoch + 1)

    logger.update(
        loss=avg_loss/len(data_loader),
        learning_rate=optimizer.param_groups[0]["lr"],
    )

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
    epoch,
    wandb,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    
    logger = {}
    val_seg_pred = {}
    
    model.eval()
    
    with tqdm.tqdm(total=len(data_loader)) as val_pbar:
        for i, batch in enumerate(data_loader):
            ims = [im.to(ptu.device) for im in batch["im"]]
            ims_metas = batch["im_metas"]
            ori_shape = ims_metas[0]["ori_shape"]
            ori_shape = (ori_shape[0].item(), ori_shape[1].item())
            filename = batch["im_metas"][0]["ori_filename"][0]

            with amp_autocast():
                seg_pred = utils.inference(
                    model_without_ddp,
                    ims,
                    ims_metas,
                    ori_shape,
                    window_size,
                    window_stride,
                    batch_size=1,
                )
                seg_pred = seg_pred.argmax(0)

            seg_pred = seg_pred.cpu().numpy()
            val_seg_pred[filename] = seg_pred
            
            val_pbar.set_description('Evaluation   ')
            val_pbar.update(1)

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )
    
    if wandb:
        WandB.log({"Val/Pixel Acc": scores["pixel_accuracy"].item() , "Val/Mean Acc": scores["mean_accuracy"].item(), "Val/Mean IoU": scores["mean_iou"].item()}, step = epoch + 1)

    for k, v in scores.items():
        logger.update(**{f"{k}": v.item()})

    return logger

def eval_dataset(
    model,
    multiscale,
    model_dir,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    save_images,
    frac_dataset,
    dataset_kwargs, wandb = False):
    
    db = create_dataset(dataset_kwargs)
    normalization = db.dataset.normalization
    dataset_name = dataset_kwargs["dataset"]
    im_size = dataset_kwargs["image_size"]
    cat_names = db.base_dataset.names
    n_cls = db.unwrapped.n_cls
    if multiscale:
        db.dataset.set_multiscale_mode()

    ims = {}
    seg_pred_maps = {}
    idx = 0
    
    with tqdm.tqdm(total = len(db)) as test_pbar:
        for i, batch in enumerate(db): 
            colors = batch["colors"]
            filename, im, seg_pred = process_batch(
                model, batch, window_size, window_stride, window_batch_size,
            )
            ims[filename] = im
            seg_pred_maps[filename] = seg_pred
            idx += 1
            if idx > len(db) * frac_dataset:
                break
            
            test_pbar.set_description('Testing     ')
            test_pbar.update(1)

    seg_gt_maps = db.dataset.get_gt_seg_maps()
    
    scores = compute_metrics(
        seg_pred_maps,
        seg_gt_maps,
        n_cls,
        ignore_index=IGNORE_LABEL,
        ret_cat_iou=True,
        distributed=ptu.distributed,
    )
    if wandb:
        WandB.log({"Test/Pixel Acc": scores["pixel_accuracy"].item() , "Test/Mean Acc": scores["mean_accuracy"].item(), "Test/Mean IoU": scores["mean_iou"].item()})

    if ptu.dist_rank == 0:
        scores["inference"] = "single_scale" if not multiscale else "multi_scale"
        suffix = "ss" if not multiscale else "ms"
        scores["cat_iou"] = np.round(100 * scores["cat_iou"], 2).tolist()
        for k, v in scores.items():
            if k != "cat_iou" and k != "inference":
                scores[k] = v.item()
            if k != "cat_iou":
                print(f"{k}: {scores[k]}")
        scores_str = yaml.dump(scores)
        with open(model_dir / f"scores_{suffix}.yml", "w") as f:
            f.write(scores_str)
        if wandb:
            WandB.save(str(model_dir) + "/" + f"scores_{suffix}.yml")
    
    if save_images:
        save_dir = model_dir / "images"
        if ptu.dist_rank == 0:
            if save_dir.exists():
                shutil.rmtree(save_dir)
            save_dir.mkdir()
        if ptu.distributed:
            torch.distributed.barrier()

        for name in sorted(ims):
            instance_dir = save_dir
            filename = name

            if dataset_name == "cityscapes":
                filename_list = name.split("/")
                instance_dir = instance_dir / filename_list[0]
                filename = filename_list[-1]
                if not instance_dir.exists():
                    instance_dir.mkdir()

            save_im(
                instance_dir,
                filename,
                ims[name],
                seg_pred_maps[name],
                torch.tensor(seg_gt_maps[name]),
                colors,
                blend,
                normalization,
            )
        if ptu.dist_rank == 0:
            shutil.make_archive(save_dir, "zip", save_dir)
            # shutil.rmtree(save_dir)
            print(f"Saved eval images in {save_dir}.zip")

    if ptu.distributed:
        torch.distributed.barrier()
        seg_pred_maps = gather_data(seg_pred_maps)