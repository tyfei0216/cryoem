import os

import mrcfile
import numpy as np
import pandas as pd
import pycocotools
import pytorch_lightning as L
import torch
import torchvision.datasets
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, Mask
from transformers import DetrForObjectDetection


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
            transforms.SanitizeBoundingBoxes(),
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
        image = torch.tensor(image * 255).type(torch.uint8)
    annotated_tensor = draw_segmentation_masks(
        image=image,
        masks=target["masks"],
        alpha=0.3,
        # colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
    )

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
            pixel_values, return_tensors="pt", pad_size={"height": 800, "width": 800}
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
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
    }


def nms(threshold, boxes, scores, classids):
    allkeep = []
    for i in np.unique(classids):
        keep = torchvision.ops.nms(
            boxes[classids == i], scores[classids == i], threshold
        )
        ids = torch.where(classids == i)[0]
        allkeep.append(ids[keep])
    keep = torch.cat(allkeep)
    return keep
