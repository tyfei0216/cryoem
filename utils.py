import os
from typing import List

import mrcfile
import numpy as np
import pandas as pd
import pycocotools
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, Mask
from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrForSegmentation,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DetrConfig,
    DetrForObjectDetection,
    DetrForSegmentation,
    DetrImageProcessor,
)

import modules


def getDefaultTransform():

    allt = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True),
            transforms.RandomResize(600, 1000),
            # transforms.Lambda(lambda x:torch.clamp(x, min=-4.0, max=4.0)),
            transforms.RandomIoUCrop(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.SanitizeBoundingBoxes(min_size=5),
        ]
    )
    return allt


def getConstantTransform():
    allt = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True),
            # transforms.RandomResize(600, 1000),
            # transforms.Lambda(lambda x:torch.clamp(x, min=-4.0, max=4.0)),
            # transforms.RandomIoUCrop(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.SanitizeBoundingBoxes(min_size=5),
        ]
    )
    return allt


def transformImage(image):
    T1 = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            transforms.RandomCrop((800, 800)),
        ]
    )
    image = T1(image)
    return image


def drawannotation(image, target):
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if np.max(image) < 10:
        image -= np.min(image)
        image /= np.max(image)
        image = torch.tensor(image * 255).type(torch.uint8)

    if "masks" in target:
        annotated_tensor = draw_segmentation_masks(
            image=image,
            masks=target["masks"],
            alpha=0.3,
            # colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
        )
    else:
        annotated_tensor = image
        annotated_tensor = torch.tensor(annotated_tensor, dtype=torch.uint8)
    # Annotate the sample image with labels and bounding boxes
    # if "names" in target:
    annotated_tensor = draw_bounding_boxes(
        image=annotated_tensor,
        boxes=target["bboxes"],
        labels=target["names"] if "names" in target else target["labels"],
        font_size=30,
        # colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
    )
    res = annotated_tensor.numpy()
    plt.imshow(np.moveaxis(res, 0, -1))


def readTomogram(filename):
    with mrcfile.open(filename, permissive=True) as m:
        return m.data


def rleUncompressed(a: np.ndarray):
    a = a.flatten()
    res = []
    t = 0
    pnt = 0
    cnt = 0
    while pnt != len(a):
        assert a[pnt] == 0 or a[pnt] == 1
        if a[pnt] == t:
            pnt += 1
            cnt += 1
        else:
            t = 1 - t
            res.append(cnt)
            cnt = 0
    return res


def get_collate_fn(image_processor):
    def collate_fn(batch):
        # DETR authors employ various image sizes during training, making it not possible
        # to directly batch together images. Hence they pad the images to the biggest
        # resolution in a given batch, and create a corresponding binary pixel_mask
        # which indicates which pixels are real/which are padding
        pixel_values = [item[0] for item in batch]
        encoding = image_processor.pad(
            pixel_values,
            return_tensors="pt",  # , pad_size={"height": 800, "width": 800}
        )
        labels = [item[1] for item in batch]
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

    return collate_fn


def stackBatch(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    labels = [item["labels"] for item in batch]
    ret = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
        # "marks": marks,
    }
    if "mark" in batch[0]:
        mark = [item["mark"] for item in batch]
    ret["mark"] = mark
    return ret


def mask_iou(mask1, mask2):
    """
    Compute IoU between two binary masks.
    """
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    return intersection / union


def donms(threshold, masks, scores):
    s = torch.zeros_like(scores, dtype=torch.float)
    seq = torch.argsort(scores, descending=True)
    keep = []
    for i in range(len(scores)):
        if s[seq[i]] < threshold:
            keep.append(seq[i])

        for j in range(i + 1, len(scores)):
            s[seq[j]] = max(s[seq[j]], mask_iou(masks[seq[i]], masks[seq[j]]))
    return torch.tensor(keep)


def nms(threshold, masks, scores, classids):
    allkeep = []
    for i in np.unique(classids):
        keep = donms(threshold, masks[classids == i], scores[classids == i])
        ids = torch.where(classids == i)[0]
        allkeep.append(ids[keep])
    keep = torch.cat(allkeep)
    return keep


def bbnms(threshold, boxes, scores, classids):
    allkeep = []
    for i in np.unique(classids):
        keep = torchvision.ops.nms(
            boxes[classids == i], scores[classids == i], threshold
        )
        ids = torch.where(classids == i)[0]
        allkeep.append(ids[keep])
    keep = torch.cat(allkeep)
    return keep


def convertBoxes(boxes):
    center_x, center_y, width, height = boxes.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [
            (center_x - 0.5 * width),
            (center_y - 0.5 * height),
            (center_x + 0.5 * width),
            (center_y + 0.5 * height),
        ],
        dim=-1,
    )
    bbox_corners[bbox_corners < 0] = 0
    bbox_corners[bbox_corners > 1] = 1
    return bbox_corners


def to_tuple(tup):
    if isinstance(tup, tuple):
        return tup
    return tuple(tup.cpu().long().tolist())


# inspired by image_processor.post_process
def postSegmentationTreatment(outputs, threshold, target_sizes, mask_threshold=0.5):
    if mask_threshold is not None:
        out_logits, out_bbox, masks = (
            outputs.logits,
            outputs.pred_boxes,
            outputs.pred_masks,
        )
    else:
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

    if target_sizes is not None:
        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    prob = nn.functional.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    boxes = convertBoxes(out_bbox)
    if isinstance(target_sizes, List):
        img_h = torch.Tensor([i[0] for i in target_sizes])
        img_w = torch.Tensor([i[1] for i in target_sizes])
        target_sizes = torch.stack([img_w, img_h], dim=1).to(boxes.device)
    else:
        img_h, img_w = target_sizes.unbind(1)

    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]

    results = []

    if mask_threshold is not None:
        for s, l, b, mask, size in zip(scores, labels, boxes, masks, target_sizes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            mask = mask[s > threshold]
            print(mask.shape, size)
            mask = nn.functional.interpolate(
                mask[:, None], size=to_tuple(size), mode="bilinear"
            ).squeeze(1)
            mask = mask.sigmoid() > mask_threshold  # * 1
            results.append(
                {"scores": score, "labels": label, "boxes": box, "masks": mask}
            )
    else:
        for s, l, b, size in zip(scores, labels, boxes, target_sizes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

    return results


def toTarget(result, size):
    if "mask" in result:
        return {
            "labels": result["labels"],
            "bboxes": BoundingBoxes(result["boxes"], format="xyxy", canvas_size=size),
            "masks": Mask(result["masks"]),
            "names": [str(i.item()) for i in result["labels"]],
        }
    else:
        return {
            "labels": result["labels"],
            "bboxes": BoundingBoxes(result["boxes"], format="xyxy", canvas_size=size),
            "names": [str(i.item()) for i in result["labels"]],
        }


def buildModel(configs, args, checkpoint=None):
    if configs["model"]["name"] == "conditional_detr":
        if configs["model"]["task"] == "segmentation":
            config = ConditionalDetrConfig(use_pretrained_backbone=False, **args)
            seg_model = ConditionalDetrForSegmentation(config)
            if len(configs["model"]["pretrained"]) > 0:
                model = ConditionalDetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
                )
                seg_model.conditional_detr.load_state_dict(model.state_dict())
            model = seg_model
        elif configs["model"]["task"] == "detection":
            if len(configs["model"]["pretrained"]) > 0:
                model = ConditionalDetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
                )
            else:
                config = ConditionalDetrConfig(use_pretrained_backbone=False, **args)
                model = ConditionalDetrForObjectDetection(config)

    elif configs["model"]["name"] == "deformable_detr":
        if len(configs["model"]["pretrained"]) > 0:
            model = DeformableDetrForObjectDetection.from_pretrained(
                configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
            )
        else:
            config = DeformableDetrConfig(use_pretrained_backbone=False, **args)
            model = DeformableDetrForObjectDetection(config)

    elif configs["model"]["name"] == "detr":
        if configs["model"]["task"] == "segmentation":
            if len(configs["model"]["pretrained"]) > 0:
                model = DetrForSegmentation.from_pretrained(
                    configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
                )
            else:
                config = DetrConfig(use_pretrained_backbone=False, **args)
                model = DetrForSegmentation(config)
        elif configs["model"]["task"] == "detection":
            if len(configs["model"]["pretrained"]) > 0:
                model = DetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"], ignore_mismatched_sizes=True, **args
                )
            else:
                config = DetrConfig(use_pretrained_backbone=False, **args)
                model = DetrForObjectDetection(config)
    else:
        raise NotImplementedError

    if checkpoint is not None and len(checkpoint) > 0:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    return model


def getModel(configs):
    models = {}
    if "checkpoint" not in configs["model"]:
        configs["model"]["checkpoint"] = None

    for i in configs["model"]["args"]:
        models[i] = buildModel(
            configs, configs["model"]["args"][i], configs["model"]["checkpoint"]
        )

    model = modules.DetrModel(
        configs["model"]["stage"],
        models,
        lr=configs["training"]["lr"],
        lr_backbone=configs["training"]["lr_backbone"],
        weight_decay=configs["training"]["weight_decay"],
        feature_dim=configs["model"]["feature_dim"],
        output_dim=configs["model"]["output_dim"],
    )
    return model
