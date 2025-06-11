import json
import os
import random
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pycocotools
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchmetrics
import torchvision.datasets
import torchvision.transforms.v2 as transforms
from PIL import Image
from pycocotools.coco import COCO
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from skimage import exposure
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as Gdataset
from torchvision.tv_tensors import BoundingBoxes, Mask
from transformers.image_transforms import center_to_corners_format

import utils


class AdditionalInputLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, in_dim)
        self.layer2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x


class EmptyContextManager:
    def __enter__(self):
        # No setup actions needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # No cleanup actions needed
        pass


from torch import Tensor


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, warmup_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
        # self.base_lr = None
        # print(self.base_lrs)

    def get_lr(self):
        # if self.base_lr is None:
        # Get the base learning rate from the optimizer
        # self.base_lr = [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_epochs:
            # print(self.last_epoch)
            # Linear warmup: Scale LR linearly based on the epoch
            return [self.warmup_lr for lr in self.base_lrs]
        else:
            # After warmup, return the base LR
            return self.base_lrs


# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(
            f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}"
        )
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(
            f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}"
        )
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


def sigmoid_focal_loss(
    inputs, targets, num_boxes=None, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if num_boxes is None:
        num_boxes = targets.shape[0]

    return loss.mean(1).sum() / num_boxes


# def focal_loss(input, target, gamma=2):
def loss_boxes(source_boxes, targets, num_boxes):
    """
    Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

    Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
    are expected in format (center_x, center_y, w, h), normalized by the image size.
    """

    loss_bbox = nn.functional.mse_loss(source_boxes, targets, reduction="none")

    # losses = {}
    # losses["loss_bbox"] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(
        generalized_box_iou(
            center_to_corners_format(source_boxes),
            center_to_corners_format(targets),
        )
    )
    # print(source_boxes[0], targets[0])
    # print(
    #     loss_bbox,
    #     loss_giou,
    # )
    loss_bbox = loss_bbox.mean()
    loss_giou = loss_giou.mean()
    # losses["loss_giou"] = loss_giou.sum() / num_boxes
    return 5 * loss_bbox + 2 * loss_giou


class DetrModel(L.LightningModule):
    """
    main model for object detection and segmentation.
    can be trained using different modes:
    stage 1: pretraining and training detr alone
    stage 2: training gnn alone
    stage mask: training mask head alone
    stage 1 + 2: training detr and gnn together
    stage 1 + 2 + 3: training detr, gnn and mask head together
    stage 1 + 2 + 3 mask: train mask head alone but with data augmentation from raw slice input

    for pretraining of detr, use stage 1
    for fine-tuning using custom data, use stage 1 + 2 + 3
    if mask is required, please further train mask head with stage mask

    all other modes are for debugging and testing purposes.

    Args:
        stage (str): the training stage, can be one of the following:
            - "stage 1": pretraining and training detr alone
            - "stage 2": training gnn alone
            - "stage mask": training mask head alone
            - "stage 1 + 2": training detr and gnn together
            - "stage 1 + 2 + 3": training detr, gnn and mask head together
            - "stage 1 + 2 + 3 mask": train mask head alone but with data augmentation from raw slice input
        model (nn.Module or dict): the detr to be trained, can be a single model or a dictionary of models.
        lr (float): learning rate for the optimizer.
        weight_decay (float): weight decay for the optimizer.
        feature_dim (int): dimension of the input features.
        output_dim (int): number of output classes.
        lr_detr (float, optional): learning rate for the DETR model (the transformer part).
        lr_backbone (float, optional): learning rate for the backbone model in the detr.
        additional_input_dim (int, optional): dimension of additional input features. Defaults to 10.
        additional_output_dim (int, optional): dimension of additional output features. Defaults to 16.
        layer_type (str, optional): type of GNN layer to use. Defaults to "GCNConv".
        dropout (bool, optional): whether to use dropout in GNN layers. Defaults to True.
        scheduler_step (int, optional): step size for the learning rate scheduler. Defaults to -1.
        warmup_epoches (int, optional): number of warmup epochs for the learning rate scheduler. Defaults to 1.
        pick_num (int, optional): number of objects to pick from each image for mask training. Defaults to 6.
        mask_alpha (float, optional): alpha value for focal loss in mask head. Defaults to 0.8.
    """

    def __init__(
        self,
        stage,
        model,
        lr,
        weight_decay,
        feature_dim,
        output_dim,
        lr_detr=None,
        lr_backbone=None,
        additional_input_dim=10,
        additional_output_dim=16,
        layer_type="GCNConv",
        dropout=True,
        scheduler_step=-1,
        warmup_epoches=1,
        pick_num=6,
        mask_alpha=0.8,
    ):
        super().__init__()
        if isinstance(model, dict):
            self.is_dict = True
            self.model = nn.ModuleDict(model)
        else:
            self.is_dict = False
            self.model = model

        assert stage in [
            "stage 1",
            "stage 2",
            "stage 1 + 2",
            "stage 1 + 2 + 3",
            "stage 1 + 2 + 3 mask",
            "stage mask",
        ]
        print("model at stage ", stage)
        self.stage = stage
        self.warmup_epoches = warmup_epoches
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.lr_detr = lr_detr
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.val_step_outputs = []
        self.additional_input_dim = additional_input_dim
        self.additional_output_dim = additional_output_dim

        # self.additional_input_layer = AdditionalInputLayer(
        #     additional_input_dim, additional_output_dim
        # )

        self.acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=output_dim, average="macro"
        )
        self.auroc = torchmetrics.AUROC(
            num_classes=output_dim, average="macro", task="multiclass"
        )
        self.mask_auroc = torchmetrics.AUROC(task="binary")

        self.gnn = GCN(
            feature_dim,
            additional_input_dim,
            output_dim,
            layer_type=layer_type,
            dropout=dropout,
        )
        t = torch.ones((output_dim))
        t[-1] = 0.1
        self.cri = nn.CrossEntropyLoss(weight=t)
        self.output_dim = output_dim
        self.mask_head = VitForMask(embed_dim=feature_dim, sigmoid=False)

        self.scheduler_step = scheduler_step

        self.num = pick_num
        self.mask_alpha = mask_alpha

        self.box_in_for_mask = True

    def forward(
        self,
        x=None,
        edge_index=None,
        pixel_values=None,
        pixel_mask=None,
        labels=None,
        mark=None,
        stage_2_embeds=None,
        box=None,
    ):
        if "stage 1" in self.stage:
            assert pixel_values is not None
            if self.is_dict:
                ret = {}
                if mark is not None:
                    ret[mark] = self.model[mark](
                        pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                    )
                else:
                    for i in self.model:
                        ret[i] = self.model[i](
                            pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels,
                        )
                return ret
            else:
                return self.model(
                    pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                )
        elif "stage 2" in self.stage:
            assert x is not None
            assert edge_index is not None
            # additional_input, model_feature = (
            #     x[:, : self.additional_input_dim],
            #     x[:, self.additional_input_dim :],
            # )
            # additional_feature = self.additional_input_layer(additional_input)
            # inputs = torch.cat([model_feature, additional_feature], dim=1)
            return self.gnn(x, edge_index)

        elif "stage mask" in self.stage:
            assert stage_2_embeds is not None
            assert pixel_values is not None
            if self.box_in_for_mask:
                assert box is not None
                return self.mask_head(
                    pixel_values,
                    stage_2_embeds,
                    box,  # .unsqueeze(1).repeat(1, self.num, 1),
                )
            else:
                return self.mask_head(pixel_values, stage_2_embeds)

        else:
            raise NotImplementedError

    def loss_boxes(self, source_boxes, targets, num_boxes):

        loss_bbox = nn.functional.mse_loss(source_boxes, targets)

        return loss_bbox  # + 2 * loss_giou

    def _common_step_stage1(self, batch, loss, loss_dict, return_outputs=False):
        if batch is None:
            return loss, loss_dict

        if "stage 1" in self.stage:

            pixel_values = batch["pixel_values"].to(self.device)
            pixel_mask = batch["pixel_mask"].to(self.device)
            if "mark" in batch:
                mark = batch["mark"][0]
            else:
                mark = None

            if mark == "":
                mark = None
            required_labels = []
            for t in batch["labels"]:
                sample = {}
                for q in ["class_labels", "boxes", "masks"]:
                    if q in t:
                        sample[q] = t[q].to(self.device)
                required_labels.append(sample)
            # labels = [
            #     {k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]
            # ]

            outputs = self(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=required_labels,
                mark=mark,
            )
            if mark is not None:
                loss += outputs[mark].loss
                if loss_dict is None:
                    loss_dict = outputs[mark].loss_dict.detach().cpu()
                else:
                    for i in loss_dict:
                        loss_dict[i] += outputs[mark].loss_dict[i].detach().cpu()
            else:
                loss = outputs.loss
                loss_dict = outputs.loss_dict
                for i in loss_dict:
                    loss_dict[i] = loss_dict[i].detach().cpu()
            if return_outputs:
                return loss, loss_dict, outputs
            return loss, loss_dict
            return loss, loss_dict
        else:
            raise ValueError("not in stage 1")

    def common_step_stage1(self, batch):
        loss = 0
        loss_dict = None
        if isinstance(batch, list):
            # print("list")
            for i in batch:
                loss, loss_dict = self._common_step_stage1(i, loss, loss_dict)
        else:
            loss, loss_dict = self._common_step_stage1(batch, loss, loss_dict)

        return loss, loss_dict

    def common_step_stage2(
        self,
        x,
        edge,
        mask,
        y,
        box=None,
        box_mask=None,
        edge_label=None,
        edge_mask=None,
        return_outputs=False,
    ):

        x = x.to(self.device)
        edge = edge.to(self.device)
        mask = mask.to(self.device)
        y = y.to(self.device)

        ret_dict = self(x=x, edge_index=edge)
        # print(ret_dict["predict"][mask].shape, y[mask].shape)
        # print(torch.max(y[mask]))

        loss = self.cri(ret_dict["predict"][mask], y[mask])
        # self.auroc.update(ret_dict["predict"][mask].detach(), y[mask])
        # print("before auroc")
        loss_dict = {
            "loss": loss.detach().cpu(),
            "auroc": self.auroc(ret_dict["predict"][mask].detach(), y[mask]).cpu(),
        }
        # print("after auroc")
        self.auroc.reset()

        if box is not None:
            box_mask = box_mask.to(self.device)
            box = box.to(self.device)
            box_mask = box_mask & mask
            loss_boxes = self.loss_boxes(
                ret_dict["box"][box_mask], box[box_mask], box_mask.sum()
            )
            loss += loss_boxes
            loss_dict["loss_boxes"] = loss_boxes.detach().cpu()
        else:
            loss_boxes = 0
        # print("before edge loss")
        if edge_label is not None:
            edge_label = edge_label.to(self.device)
            if edge_mask is None:
                edge_mask = torch.ones_like(edge_label, dtype=torch.bool)
            loss_edge = F.binary_cross_entropy(
                ret_dict["edge"].squeeze()[edge_mask], edge_label[edge_mask].float()
            )
            loss += loss_edge
            loss_dict["loss_edge"] = loss_edge.detach().cpu()
        # print("after edge loss")

        if return_outputs:
            return loss, loss_dict, ret_dict
        return loss, loss_dict

    def common_stage_mask(
        self, pixel_values, stage_2_embeds, mask, cal_auroc=False, box=None
    ):
        pixel_values = pixel_values.to(self.device).float()
        stage_2_embeds = stage_2_embeds.to(self.device).float()
        mask = mask.to(self.device).float()
        outputs = self(
            pixel_values=pixel_values, stage_2_embeds=stage_2_embeds, box=box
        )
        loss = sigmoid_focal_loss(outputs, mask.float(), alpha=self.mask_alpha)
        # print("after focal loss")
        if cal_auroc:
            auroc = self.mask_auroc(
                outputs.detach().view(-1).cpu(), mask.detach().view(-1).cpu()
            ).cpu()
            # self.mask_auroc.reset()
            # torch.cuda.empty_cache()
            return loss, {"loss": loss.detach().cpu(), "mask_auroc": auroc}
        # loss = F.binary_cross_entropy(outputs, mask.float())
        return loss, {"loss": loss.detach().cpu()}

    def on_validation_epoch_end(self):
        # loss = torch.stack(self.val_step_outputs).mean()
        losses = np.mean([i["loss"] for i in self.val_step_outputs])
        if "stage mask" in self.stage or "stage 1 + 2 + 3" in self.stage:
            # auroc = self.mask_auroc.compute()
            auroc = np.nanmean(
                [i["mask_auroc"] for i in self.val_step_outputs if "mask_auroc" in i]
            )
            self.log("total_validate_mask_auroc", auroc, prog_bar=True)
            self.mask_auroc.reset()
        if "stage 2" in self.stage or "stage 1 + 2" in self.stage:
            # auroc = self.auroc.compute()
            auroc = np.mean([i["auroc"] for i in self.val_step_outputs])
            self.log("total_validate_auroc", auroc, prog_bar=True)
            self.auroc.reset()
            loss_boxes = np.mean(
                [i["loss_boxes"] for i in self.val_step_outputs if "loss_boxes" in i]
            )
            self.log("total_validate_loss_boxes", loss_boxes, prog_bar=True)

        self.log("total_validate_loss", losses, prog_bar=True)
        self.val_step_outputs.clear()

    def on_train_epoch_end(self):
        # loss = torch.stack(self.val_step_outputs).mean()
        losses = np.mean([i["loss"] for i in self.training_step_outputs])
        if "stage mask" in self.stage or "stage 1 + 2 + 3" in self.stage:
            auroc = np.nanmean(
                [
                    i["mask_auroc"]
                    for i in self.training_step_outputs
                    if "mask_auroc" in i
                ]
            )
            self.log("total_train_mask_auroc", auroc, prog_bar=True)
            self.mask_auroc.reset()

        if "stage 2" in self.stage or "stage 1 + 2" in self.stage:
            # auroc = self.auroc.compute()
            auroc = np.mean([i["auroc"] for i in self.training_step_outputs])
            self.log("total_train_auroc", auroc, prog_bar=True)
            self.auroc.reset()
            loss_boxes = np.mean(
                [
                    i["loss_boxes"]
                    for i in self.training_step_outputs
                    if "loss_boxes" in i
                ]
            )
            self.log("total_train_loss_boxes", loss_boxes, prog_bar=True)

        self.log("total_train_loss", losses, prog_bar=True)
        self.training_step_outputs.clear()

    def _common_step(self, batch):
        if "stage 1 + 2 + 3" in self.stage:
            temp = self.stage
            n, _, _, _ = batch[0]["pixel_values"].shape

            t = torch.no_grad
            if self.lr_detr > 0.0000001:
                t = EmptyContextManager
            # print("stage 1")
            with t():
                loss, loss_dict, output = self._common_step_stage1(
                    batch[0], 0, None, True
                )
                # print("stage 2")
                retdict = utils.process(output, batch[0]["labels"], need_mask=True)
                data2 = utils.convertStage2Dataset(retdict)
                self.stage = "stage 2"
                x = data2.x
                y = data2.y
                edge_index = data2.edge_index
                mask = torch.ones_like(y, dtype=torch.bool)
                boxes = data2.boxes  # if hasattr(data2, "boxes") else None
                box_masks = data2.box_masks  # if hasattr(data2, "box_masks") else None
                # if box_masks is not None:
                #     box_masks = box_masks>-1
                edge_label = (
                    data2.edge_label
                )  # if hasattr(batch, "edge_label") else None
                # edge_label = None
                # if edge_label is None:
                #     print("edge_label is None")
                loss2, loss_dict2, outputs = self.common_step_stage2(
                    x,
                    edge_index,
                    mask,
                    y,
                    boxes,
                    box_masks > -1,
                    edge_label,
                    return_outputs=True,
                )
                loss += loss2
                self.stage = "stage mask"

                img = batch[0]["pixel_values"][n // 2]
                embeds = outputs["embeddings"]
                objects, _ = embeds.shape
                obj_per_image = objects // n
                masks = []
                stage_2_embeds = []
                sub_embeds = embeds[
                    (n // 2) * obj_per_image : (n // 2 + 1) * obj_per_image
                ]
                sub_box_masks = box_masks[
                    (n // 2) * obj_per_image : (n // 2 + 1) * obj_per_image
                ]
                # y = data2.y
                # y = y[(n // 2) * obj_per_image : (n // 2 + 1) * obj_per_image]
                # t = retdict["masks"]
                pick_from = torch.where((sub_box_masks > -1))[0]
                boxes = outputs["box"]

            if len(pick_from) > 0:
                if len(pick_from) <= self.num:
                    stage_2_embeds = sub_embeds[pick_from]
                    pick_index = sub_box_masks[pick_from]
                    box = boxes[pick_from]
                    masks = retdict["masks"][pick_index]

                else:
                    tensor = torch.arange(len(pick_from))
                    indices = torch.randperm(tensor.size(0))[: self.num]
                    selected = pick_from[indices]
                    stage_2_embeds = sub_embeds[selected]
                    box = boxes[selected]
                    pick_index = sub_box_masks[selected]
                    masks = retdict["masks"][pick_index]
                num_masks = masks.sum(axis=[1, 2])
                num_masks = num_masks > 0
                if num_masks.sum() > 0:
                    stage_2_embeds = stage_2_embeds[num_masks]
                    masks = masks[num_masks]
                    box = box[num_masks]
                    img = img[None, :, :, :].repeat(stage_2_embeds.shape[0], 1, 1, 1)
                    loss3, loss_dict3 = self.common_stage_mask(
                        img, stage_2_embeds, masks, True, box
                    )
                    loss += loss3
                    loss_dict2["mask_auroc"] = loss_dict3["mask_auroc"]
            else:
                loss_dict2["mask_auroc"] = np.nan

            self.stage = temp  # "stage 1 + 2 + 3"
            loss_dict = loss_dict2
            # loss_dict["auroc_mask"] = loss_dict3["auroc"]
        elif "stage 1 + 2" in self.stage:
            loss, loss_dict, output = self._common_step_stage1(batch[0], 0, None, True)
            retdict = utils.process(output, batch[0]["labels"])
            data2 = utils.convertStage2Dataset(retdict)
            self.stage = "stage 2"
            x = data2.x
            y = data2.y
            edge_index = data2.edge_index
            mask = torch.ones_like(y, dtype=torch.bool)
            boxes = data2.boxes if hasattr(data2, "boxes") else None
            box_masks = data2.box_masks if hasattr(data2, "box_masks") else None
            edge_label = data2.edge_label  # if hasattr(batch, "edge_label") else None
            # edge_label = None
            if edge_label is None:
                print("edge_label is None")
            loss2, loss_dict2 = self.common_step_stage2(
                x, edge_index, mask, y, boxes, box_masks, edge_label
            )
            self.stage = "stage 1 + 2"
            loss = loss + loss2
            loss_dict = loss_dict2
        elif "stage 1" in self.stage:
            loss, loss_dict = self.common_step_stage1(batch)
        elif "stage 2" in self.stage:
            mask = batch.train_mask
            y = batch.y
            boxes = batch.boxes if hasattr(batch, "boxes") else None
            box_masks = batch.box_masks if hasattr(batch, "box_masks") else None
            edge_label = batch.edge_label if hasattr(batch, "edge_label") else None
            edge_mask = (
                batch.train_edge_mask if hasattr(batch, "train_edge_mask") else None
            )
            loss, loss_dict = self.common_step_stage2(
                batch.x,
                batch.edge_index,
                mask,
                y,
                boxes,
                box_masks,
                edge_label,
                edge_mask,
            )

        elif "stage mask" in self.stage:

            pixel_values, stage_2_embeds, pixel_mask, box = batch
            # print(pixel_mask)
            loss, loss_dict = self.common_stage_mask(
                pixel_values, stage_2_embeds, pixel_mask, True, box=box
            )

        return loss, loss_dict

    def training_step(self, batch, batch_idx=0, loader_idx=0):

        loss, loss_dict = self._common_step(batch)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss, prog_bar=True)
        # for k, v in loss_dict.items():
        #     self.log("train_" + k, v.item(), prog_bar=False)
        self.training_step_outputs.append(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx=0, loader_idx=0):
        # print(batch)
        loss, loss_dict = self._common_step(batch)
        res = {}
        res["loss"] = loss.detach().cpu()
        for k, v in loss_dict.items():
            res[k] = v.detach().cpu()
        self.val_step_outputs.append(res)
        self.log("validation_loss", res["loss"], prog_bar=True)
        # for k, v in loss_dict.items():
        #     self.log("validate_" + k, v.item(), prog_bar=False)

        return loss

    def configure_optimizers(self):
        optim = None
        if "stage 1 + 2 + 3 mask" in self.stage:
            d1 = []
            d2 = []
            for n, p in self.named_parameters():
                if "mask_head" in n and p.requires_grad:
                    d1.append(p)
                elif p.requires_grad:
                    d2.append(p)

            param_dicts = [
                {
                    "params": d1,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
            ]
            if self.lr_detr > 0.0000001:
                param_dicts.append(
                    {
                        "params": d2,
                        "lr": self.lr_detr,
                        "weight_decay": self.weight_decay * 0.01,
                    },
                )
            optim = torch.optim.AdamW(param_dicts)

        elif "stage 1" in self.stage:
            if self.lr_backbone is not None:

                d1 = []
                d2 = []
                d3 = []
                for n, p in self.named_parameters():
                    if "backbone" in n and p.requires_grad:
                        d1.append(p)
                    elif ".model" in n:
                        d2.append(p)
                    else:
                        d3.append(p)
                # self.lr_backbone = self.lr
                param_dicts = [
                    {"params": d1, "lr": self.lr_backbone},
                    {
                        "params": d2,
                        "lr": self.lr_detr,
                    },
                    {
                        "params": d3,
                        "lr": self.lr,
                    },
                ]
                if self.weight_decay > 0:
                    optim = torch.optim.AdamW(
                        param_dicts, weight_decay=self.weight_decay
                    )
                else:
                    optim = torch.optim.Adam(param_dicts)

        elif "stage 2" in self.stage:
            parameters = []
            for n, p in self.named_parameters():
                if ".model" in n or "mask_head" in n:
                    p.requires_grad = False
                else:
                    parameters.append(p)

            if self.weight_decay > 0:
                optim = torch.optim.AdamW(
                    parameters, lr=self.lr, weight_decay=self.weight_decay
                )
            else:
                optim = torch.optim.Adam(parameters, lr=self.lr)
        elif "stage mask" in self.stage:
            print("fixing all parameters except mask head")
            parameters = []
            for n, p in self.named_parameters():
                if not "mask_head" in n:
                    p.requires_grad = False
                else:
                    parameters.append(p)

            if self.weight_decay > 0:
                optim = torch.optim.AdamW(
                    parameters, lr=self.lr, weight_decay=self.weight_decay
                )
            else:
                optim = torch.optim.Adam(parameters, lr=self.lr)
        if optim is None:
            if self.weight_decay > 0:
                optim = torch.optim.AdamW(
                    self.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            else:
                optim = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler_step > 0:
            return [optim], [
                {
                    "scheduler": WarmupScheduler(optim, self.warmup_epoches),
                    "interval": "epoch",
                    "frequency": 1,
                },
                {
                    "scheduler": torch.optim.lr_scheduler.StepLR(
                        optim, step_size=1, gamma=0.5
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                },
            ]
        else:
            return optim


import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    AGNNConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    SAGEConv,
    TransformerConv,
)


class SimpleLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge):
        return self.linear(x)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, dtype=torch.float):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).to(dtype))
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank, dtype=dtype) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim, dtype=dtype) * std_dev)
        # self.C = torch.nn.Parameter(torch.zeros(out_dim, dtype=dtype) * std_dev)
        # self.alpha = alpha

    def forward(self, x):
        x = x @ self.A @ self.B  # + self.C
        # print(x.shape)
        # x = x + self.C
        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        additional_input_dim,
        output_classes,
        layer_type="GCNConv",
        dropout=False,
        zpos=50,
    ):

        super().__init__()
        self.output_classes = output_classes
        self.additional_input_dim = additional_input_dim
        self.additional_input_layer = AdditionalInputLayer(
            additional_input_dim, input_dim
        )
        if layer_type == "GCNConv":
            Layer = GCNConv
        elif layer_type == "SAGEConv":
            Layer = SAGEConv
        elif layer_type == "GATConv":
            Layer = GATConv
        elif layer_type == "AGNNConv":
            Layer = AGNNConv
        elif layer_type == "GATv2Conv":
            Layer = GATv2Conv
        elif layer_type == "TransformerConv":
            Layer = TransformerConv
        elif layer_type == "SimpleLinear":
            Layer = SimpleLinear
        else:
            raise ValueError("Invalid layer type")

        pe = self.inipos(input_dim)
        pe.requires_grad = False
        self.register_buffer("pe", pe)

        self.conv1 = Layer(input_dim, input_dim)
        self.conv2 = Layer(input_dim, input_dim)
        self.conv3 = Layer(input_dim, input_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_classes),
        )
        # self.cls_head = nn.Sequential(
        #     nn.Linear(input_dim, input_dim // 2),
        #     nn.ReLU(),
        #     LoRALayer(input_dim // 2, output_classes, 4),
        # )
        self.box_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            LoRALayer(input_dim // 2, 4, 4),
            nn.Tanh(),
        )

        self.dropout = dropout

        self.edge_head = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.zpos = zpos

    def inipos(self, channels):
        inv_freq = 1.0 / (
            (55 * 10) ** (torch.arange(0, channels, 2).float() / channels)
        )  # .to(self.device)
        t = torch.arange(0, 55)[:, None]  # .to(self.device)
        # print(t.shape, inv_freq.shape)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # print(pos_enc_a.shape)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
        return pos_enc

    def forward(self, inputs, edge_index):

        additional_input, model_feature = (
            inputs[:, : self.additional_input_dim],
            inputs[:, self.additional_input_dim :],
        )

        pos = (additional_input[:, 0] * self.zpos).long()
        pos_embed = self.pe[pos]

        feature = self.additional_input_layer(inputs)
        x = feature + pos_embed
        # x = torch.cat([pos_embed, feature], dim=1)
        # if self.dropout:
        #     model_feature = F.dropout(model_feature, p=0.2)
        # x = torch.cat([model_feature, additional_feature], dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=0.2)
        x = self.conv2(x, edge_index)

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=0.2)
        x = self.conv3(x, edge_index)

        # if self.dropout:
        #     x = F.dropout(x, p=0.2)

        predict = self.cls_head(x)
        # predict[:, : self.output_classes - 1] += inputs[:, 5 : self.output_classes + 4]

        box = self.box_head(x) + inputs[:, 1:5]

        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=1)
        edge = self.edge_head(edge_embeddings)

        return {"predict": predict, "box": box, "edge": edge, "embeddings": x}


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return F.gelu(self.double_conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, pool_kernal=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_kernal),
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(emb_dim, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, size, emb_dim=None):
        super().__init__()

        self.up = nn.Upsample(size=size, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels),
        )
        if emb_dim is not None:
            self.emb_layer = nn.Sequential(
                nn.Linear(emb_dim, out_channels),
                nn.SiLU(),
            )

    def forward(self, x, skip_x, t=None):
        x = self.up(x)
        # print("up", x.shape, skip_x.shape)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        if t is None:
            return x

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class VitForMask(nn.Module):
    def __init__(self, c_in=3, c_out=1, embed_dim=272, sigmoid=True):
        super().__init__()
        # self.ini = DoubleConv()

        # 16 * 800 * 800
        self.inc = DoubleConv(c_in + 1, 16)

        # 16 * 400 * 400
        self.down1 = Down(16, 32, embed_dim)

        # 64 * 200 * 200
        self.down2 = Down(32, 64, embed_dim)

        # 128 * 100 * 100
        self.down3 = Down(64, 128, embed_dim)

        # 256 * 50 * 50
        self.down4 = Down(128, 512, embed_dim)

        # 512 * 25 * 25
        # self.down5 = Down(256, 512, embed_dim)
        self.l = nn.Sequential(nn.Linear(embed_dim, 512), nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(1, 2500 + 1, 512))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu",
            ),
            num_layers=6,
        )

        # 128 * 100 * 100
        self.up3 = Up(512 + 128, 64, (100, 100), embed_dim)

        # 64 * 200 * 200
        self.up4 = Up(128, 32, (200, 200), embed_dim)

        # 32 * 400 * 400
        self.up5 = Up(64, 16, (400, 400), embed_dim)

        # 32 * 800 * 800
        self.up6 = Up(32, 32, (800, 800), embed_dim)

        # 1 * 800 * 800
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        self.sigmoid = sigmoid

    def forward(self, x, t, boxes):
        B, C, H, W = x.shape
        b1 = boxes[:, 0] - boxes[:, 2] / 2
        b2 = boxes[:, 0] + boxes[:, 2] / 2
        b3 = boxes[:, 1] - boxes[:, 3] / 2
        b4 = boxes[:, 1] + boxes[:, 3] / 2
        b1, b2, b3, b4 = (
            (b1 * W).long(),
            (b2 * W).long(),
            (b3 * H).long(),
            (b4 * H).long(),
        )
        mask = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
        for i in range(B):
            mask[i, 0, b3[i] : b4[i], b1[i] : b2[i]] = 1.0

        mask.requires_grad_(False)

        x = torch.cat((x, mask), dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)

        x5 = x5.view(-1, 512, 2500).transpose(1, 2)
        l = self.l(t).unsqueeze(1)
        x = torch.cat((l, x5), dim=1)
        x += self.pos_embed
        # x5 = x5.transpose()
        x = self.transformer(x)
        x = x[:, 1:, :]
        x = x.transpose(1, 2).view(-1, 512, 50, 50)

        x = self.up3(x, x4, t)
        x = self.up4(x, x3, t)
        x = self.up5(x, x2, t)
        x = self.up6(x, x1, t)
        output = self.outc(x).squeeze(1)

        if self.sigmoid:
            output = torch.sigmoid(output)

        return output
