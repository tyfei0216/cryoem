import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pycocotools
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms.v2 as transforms
from PIL import Image
from pycocotools.coco import COCO
from torchvision.tv_tensors import BoundingBoxes, Mask

import utils


class MyCOCO(COCO):
    def __init__(self, annotation_file=None):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print("loading annotations into memory...")
            tic = time.time()
            if annotation_file.endswith(".json"):
                with open(annotation_file, "r") as f:
                    dataset = json.load(f)
            elif annotation_file.endswith(".pkl"):
                import pickle

                with open(annotation_file, "rb") as f:
                    dataset = pickle.load(f)
            else:
                raise NotImplementedError
            assert (
                type(dataset) == dict
            ), "annotation file format {} not supported".format(type(dataset))
            print("Done (t={:0.2f}s)".format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()


class CocoDetection(torchvision.datasets.vision.VisionDataset):

    def __init__(
        self,
        image_directory_path: str,
        annotation_file_path: str,
        is_npy=True,
        require_mask=False,
        filter_class=None,
        single_class=False,
        transform=None,
        add_classname=False,
        clamp=[-4.0, 4.0],
        maxsize=800,
        return_single=None,
        mark="",
    ):
        # annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super().__init__(image_directory_path)
        # super().__init__(image_directory_path, annotation_file_path)
        self.coco = MyCOCO(annotation_file_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.mark = mark

        self.is_npy = is_npy
        self.require_mask = require_mask
        self.filter_class = filter_class
        self.single_class = single_class
        if self.filter_class is not None:
            if self.single_class:
                self.mapping_class = {i: 0 for i in filter_class}
            else:
                self.mapping_class = {i: idx for idx, i in enumerate(filter_class)}
        else:
            self.mapping_class = None

        self.transform = transform
        self.classes = pd.DataFrame(self.coco.dataset["categories"])
        self.classes = self.classes.set_index("id")
        self.add_classname = add_classname
        self.clamp = clamp
        self.maxsize = maxsize

        self.return_single = return_single
        self._filterIds()

    def _load_image(self, id: int):
        if self.is_npy:
            path = self.coco.loadImgs(id)[0]["file_name"]
            path = os.path.join(self.root, path)
            return np.load(path, allow_pickle=True)
        else:
            path = self.coco.loadImgs(id)[0]["file_name"]
            return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        if self.filter_class is None:
            return self.coco.loadAnns(self.coco.getAnnIds(id))
        else:
            t = self.coco.loadAnns(self.coco.getAnnIds(id))
            t = list(filter(lambda x: x["category_id"] in self.filter_class, t))
            return t

    def processAnnotations(self, annotations, image):
        # print(image.shape)
        if self.mapping_class is not None:
            labels = torch.tensor(
                [self.mapping_class[sample["category_id"]] for sample in annotations],
                dtype=torch.long,
            )
        else:
            labels = torch.tensor(
                [sample["category_id"] for sample in annotations], dtype=torch.long
            )
        if self.single_class:
            labels = torch.zeros_like(labels)
        # labels = [str(sample["category_id"]) for sample in annotations]
        bbboxes = torch.stack(
            [torch.tensor(mask["bbox"]) for mask in annotations], dim=0
        )

        bt = torchvision.ops.box_convert(torch.Tensor(bbboxes), "xywh", "xyxy")
        boxes = BoundingBoxes(bt, format="xyxy", canvas_size=image.shape[1:])

        if self.require_mask:
            masks = torch.stack(
                [
                    torch.tensor(
                        pycocotools.mask.decode(mask["segmentation"]), dtype=torch.bool
                    )
                    for mask in annotations
                ],
                dim=0,
            )
            return {"masks": Mask(masks), "bboxes": boxes, "class_labels": labels}
        else:
            return {"bboxes": boxes, "class_labels": labels}

    def __len__(self):
        return len(self.needids)

    def _filterIds(self):
        needids = []
        for i in range(len(self.ids)):
            _, annotation = self._getitem(i)
            if len(annotation) > 0:
                needids.append(i)
            # if len(self.coco.loadAnns(self.coco.getAnnIds(i))) > 0:
            #     need.append(i)
        self.needids = needids

    def _getitem(self, index):
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        return image, target

    def __getitem__(self, idx):
        if self.needids is not None:
            idx = self.needids[idx]

        # used to return a single image, used for debugging
        if self.return_single is not None:
            idx = self.return_single

        # image, annotation = super().__getitem__(idx)
        image, annotation = self._getitem(idx)
        if len(annotation) == 0:
            raise ValueError
        image = torch.tensor(image)
        target = self.processAnnotations(annotation, image)

        if self.transform is not None:
            image, target = self.transform(image, target)

        c, h, w = image.shape
        target["orig_size"] = torch.tensor((h, w))
        if h > self.maxsize or w > self.maxsize:
            h = h if h < self.maxsize else self.maxsize
            w = w if w < self.maxsize else self.maxsize
            temp = transforms.Compose(
                [
                    transforms.Resize((min(h, self.maxsize), min(w, self.maxsize))),
                    transforms.SanitizeBoundingBoxes(),
                ]
            )
            image, target = temp(image, target)

        mask = torch.zeros((self.maxsize, self.maxsize), dtype=torch.long)
        mask[:h, :w] = 1

        padtransform = transforms.Pad(
            (0, 0, self.maxsize - w, self.maxsize - h), fill=0
        )
        image, target = padtransform(image, target)

        target["size"] = torch.tensor((self.maxsize, self.maxsize))
        target["image_id"] = torch.tensor((idx))

        target["boxes"] = torch.zeros_like(target["bboxes"], dtype=torch.float32)
        target["boxes"][:, 0] = (target["bboxes"][:, 0] + target["bboxes"][:, 2]) / (
            self.maxsize * 2
        )
        target["boxes"][:, 1] = (target["bboxes"][:, 1] + target["bboxes"][:, 3]) / (
            self.maxsize * 2
        )
        target["boxes"][:, 2] = (
            -target["bboxes"][:, 0] + target["bboxes"][:, 2]
        ) / self.maxsize
        target["boxes"][:, 3] = (
            -target["bboxes"][:, 1] + target["bboxes"][:, 3]
        ) / self.maxsize

        if self.require_mask:
            target["area"] = target["masks"].sum([1, 2])
        else:
            target["area"] = (
                target["boxes"][:, 2]
                * target["boxes"][:, 3]
                * self.maxsize
                * self.maxsize
            )

        target["iscrowd"] = torch.zeros(
            (len(target["class_labels"])), dtype=torch.int64
        )

        if self.add_classname:
            target["names"] = []
            for i in target["class_labels"]:
                target["names"].append(self.classes.loc[i.item()]["name"])

        return {
            "pixel_values": image,
            "pixel_mask": mask,
            "labels": target,
            "mark": self.mark,
        }


# Deprecated
class MyCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        annotation_file_path: str,
        image_processor=None,
        is_npy=True,
        filter_class=None,
        transform=None,
        return_single=None,
    ):
        # annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super().__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor
        self.is_npy = is_npy
        self.filter_class = filter_class
        self.transform = transform
        self.return_single = return_single

    def _load_image(self, id: int):
        if self.is_npy:
            path = self.coco.loadImgs(id)[0]["file_name"]
            path = os.path.join(self.root, path)
            return np.load(path, allow_pickle=True)
        else:
            return super()._load_image(id)

    def _load_target(self, id: int):
        if self.filter_class is None:
            return self.coco.loadAnns(self.coco.getAnnIds(id))
        else:
            t = self.coco.loadAnns(self.coco.getAnnIds(id))
            t = list(filter(lambda x: x["category_id"] in self.filter_class, t))
            return t

    def __getitem__(self, idx):
        if self.return_single is not None:
            idx = self.return_single
        images, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {"image_id": image_id, "annotations": annotations}
        encoding = self.image_processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
            return_segmentation_masks=True,
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target

        # return images, annotations


class DetrModel(L.LightningModule):
    def __init__(
        self,
        stage,
        model_dict,
        lr,
        weight_decay,
        feature_dim,
        output_dim,
        lr_backbone=None,
    ):
        super().__init__()
        self.model = nn.ModuleDict(model_dict)

        assert stage in ["stage 1", "stage 2"]
        self.stage = stage

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.val_step_outputs = []

        self.gnn = GCN(feature_dim, output_dim)

    def forward(self, pixel_values, pixel_mask, labels=None, mark=None):
        ret = {}
        if mark is not None:
            ret[mark] = self.model[mark](
                pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
            )
        else:
            for i in self.model:
                ret[i] = self.model[i](
                    pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
                )
        return ret

    def _common_step(self, batch, loss, loss_dict):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        mark = batch["mark"][0]

        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, mark=mark
        )

        loss += outputs[mark].loss
        if loss_dict is None:
            loss_dict = outputs[mark].loss_dict
        else:
            for i in loss_dict:
                loss_dict[i] += outputs[mark].loss_dict[i]

        return loss, loss_dict

    def common_step(self, batch):
        loss = 0
        loss_dict = None
        if isinstance(batch, list):
            print("list")
            for i in batch:
                loss, loss_dict = self._common_step(i, loss, loss_dict)
        else:
            loss, loss_dict = self._common_step(batch, loss, loss_dict)

        return loss, loss_dict

    def on_validation_epoch_end(self):
        loss = torch.stack(self.val_step_outputs).mean()
        loss = np.mean(self.val_step_outputs)
        self.log("validate_loss", loss)
        self.val_step_outputs.clear()

    def training_step(self, batch, dataloader_idx=0, dataloader_iter=0):
        loss, loss_dict = self.common_step(batch)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, dataloader_idx=0, dataloader_iter=0):
        loss, loss_dict = self.common_step(batch)
        self.val_step_outputs.append(loss.detach().cpu())
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        if self.lr_backbone is not None:
            # self.lr_backbone = self.lr
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if "backbone" not in n and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if "backbone" in n and p.requires_grad
                    ],
                    "lr": self.lr_backbone,
                },
            ]
            return torch.optim.AdamW(
                param_dicts, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            return torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )


# only used for pretraining
class Detr(L.LightningModule):

    def __init__(self, lr, weight_decay, model, lr_backbone=None):
        super().__init__()
        self.model = model

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.val_step_outputs = []

    def forward(self, pixel_values, pixel_mask, labels=None):

        # pixel_values = pixel_values.to(self.device)
        # pixel_mask = pixel_mask.to(self.device)
        if labels is not None:
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
            outputs = self.model(
                pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
            )
        else:
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    # def forward(self, pixel_values, pixel_mask):
    #     return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def on_validation_epoch_end(self):
        loss = torch.stack(self.val_step_outputs).mean()
        loss = np.mean(self.val_step_outputs)
        self.log("validate_loss", loss)
        self.val_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.val_step_outputs.append(loss.detach().cpu())
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        if self.lr_backbone is not None:
            # self.lr_backbone = self.lr
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if "backbone" not in n and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if "backbone" in n and p.requires_grad
                    ],
                    "lr": self.lr_backbone,
                },
            ]
            return torch.optim.AdamW(
                param_dicts, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            return torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )


class EMDataModule(L.LightningDataModule):
    def __init__(self, trainsets, valsets, train_batch_size, val_batch_size):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.trainset = trainsets
        self.valset = valsets

    def train_dataloader(self):
        train_sets = []
        for i in self.trainset:
            train_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.trainset[i],
                    collate_fn=utils.stackBatch,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                )
            )
        return train_sets

    def val_dataloader(self):
        val_sets = []
        for i in self.valset:
            val_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.valset[i],
                    collate_fn=utils.stackBatch,
                    batch_size=self.val_batch_size,
                )
            )
        return val_sets

    def test_dataloader(self):
        return self.testset


# from transformers import DetrForObjectDetection, DetrForSegmentation


# original Detr class
class _Detr(L.LightningModule):
    def __init__(self, model, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.model = model
        # DetrForObjectDetection.from_pretrained(
        #     "facebook/detr-resnet-50",
        #     revision="no_timm",
        #     num_labels=len(id2label),
        #     ignore_mismatched_sizes=True,
        # )
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask, labels=None):

        pixel_values = pixel_values.to(self.device)
        pixel_mask = pixel_mask.to(self.device)
        if labels is not None:
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
            outputs = self.model(
                pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
            )
        else:
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optim = torch.optim.Adam(
            param_dicts, lr=self.lr  # , weight_decay=self.weight_decay
        )
        return optim
        # optimizer = torch.optim.AdamW(
        #     param_dicts, lr=self.lr, weight_decay=self.weight_decay
        # )

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_classes):
        super().__init__()
        self.conv1 = GCNConv(input_dim, input_dim // 2)
        self.conv2 = GCNConv(input_dim // 2, input_dim // 4)
        self.conv3 = GCNConv(input_dim // 4, output_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.conv2(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
