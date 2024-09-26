import os

import numpy as np
import pandas as pd
import pycocotools
import pytorch_lightning as L
import torch
import torchvision.datasets
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, Mask


class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(
        self,
        image_directory_path: str,
        annotation_file_path: str,
        is_npy=True,
        require_mask=False,
        filter_class=None,
        transform=None,
        add_classname=False,
        clamp=[-4.0, 4.0],
        maxsize=800,
    ):
        # annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super().__init__(image_directory_path, annotation_file_path)
        self.is_npy = is_npy
        self.require_mask = require_mask
        self.filter_class = filter_class
        self.transform = transform
        self.classes = pd.DataFrame(self.coco.dataset["categories"])
        self.classes = self.classes.set_index("id")
        self.add_classname = add_classname
        self.clamp = clamp
        self.maxsize = maxsize

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
            t = list(filter(lambda x: x["category_id"] in self.filter_class), t)
            return t

    def processAnnotations(self, annotations, image):
        # print(image.shape)
        labels = torch.tensor(
            [sample["category_id"] for sample in annotations], dtype=torch.long
        )
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

    def __getitem__(self, idx):
        image, annotation = super().__getitem__(idx)
        if len(annotation) == 0:
            raise ValueError
        image = torch.tensor(image)
        target = self.processAnnotations(annotation, image)

        if self.transform is not None:
            image, target = self.transform(image, target)

        c, h, w = image.shape
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

        if self.add_classname:
            target["names"] = []
            for i in target["class_labels"]:
                target["names"].append(self.classes.loc[i.item()]["name"])

        return {"pixel_values": image, "pixel_mask": mask, "labels": target}


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
    ):
        # annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super().__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor
        self.is_npy = is_npy
        self.filter_class = filter_class
        self.transform = transform

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
            t = list(filter(lambda x: x["category_id"] in self.filter_class), t)
            return t

    def __getitem__(self, idx):
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


class Detr(L.LightningModule):

    def __init__(self, lr, weight_decay, model, lr_backbone=None):
        super().__init__()
        self.model = model

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.val_step_outputs = []

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

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
        self.log("training_loss", loss)
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
        if self.lr_backbone is None:
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
    def __init__(self, trainset, valset, testset):
        super().__init__()
        self.trainset = trainset
        self.testset = testset
        self.valset = valset

    def train_dataloader(self):
        return self.trainset

    def val_dataloader(self):
        return self.valset

    def test_dataloader(self):
        return self.testset
