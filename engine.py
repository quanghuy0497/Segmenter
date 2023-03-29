import torch
import math
import tqdm
import utils.torch as ptu
import wandb as WandB

from utils.logger import MetricLogger
from metrics import gather_data, compute_metrics
from model import utils
from data.utils import IGNORE_LABEL



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
