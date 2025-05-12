import json
import os
import pickle
import random
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


class stage2Dataset(Gdataset):
    def __init__(self, dataset_list, dataset_len, train_val=[0.8, 0.2], seed=1013):
        super().__init__()
        self.dataset_list = []
        self.dataset_len = dataset_len
        random.seed(seed)
        for i in dataset_list:
            a = list(range(i.x.shape[0]))
            train = random.sample(a, int(len(a) * train_val[0]))
            train_mask = torch.zeros(len(a), dtype=torch.bool)
            train_mask[train] = True
            val_mask = ~train_mask
            i.train_mask = train_mask
            i.val_mask = val_mask
            if hasattr(i, "edge_label"):
                edge_train_mask = torch.zeros(i.edge_label.shape[0], dtype=torch.bool)
                train = random.sample(
                    list(range(i.edge_label.shape[0])), int(len(a) * train_val[0])
                )
                edge_train_mask[train] = True
                i.edge_train_mask = edge_train_mask
                i.edge_val_mask = ~edge_train_mask
            self.dataset_list.append(i)

    def len(self):
        return self.dataset_len

    def get(self, idx):
        t = idx % len(self.dataset_list)
        return self.dataset_list[t]


class stage2DatasetNew(Gdataset):
    def __init__(self, dataset_list, dataset_len, train=True, seed=1013, aug=True):
        super().__init__()
        self.dataset_list = []
        self.dataset_len = dataset_len
        random.seed(seed)
        self.if_train = train
        self.aug = aug
        for i in dataset_list:
            a = list(range(i.x.shape[0]))
            # train = random.sample(a, int(len(a) * train_val[0]))
            train_mask = torch.zeros(len(a), dtype=torch.bool)
            val_mask = torch.zeros(len(a), dtype=torch.bool)
            if train:
                train_mask[...] = True
            else:
                val_mask[...] = True
            # train_mask[train] = True
            # val_mask = ~train_mask
            i.train_mask = train_mask
            i.val_mask = val_mask
            if hasattr(i, "edge_label"):
                edge_train_mask = torch.zeros(i.edge_label.shape[0], dtype=torch.bool)
                edge_val_mask = torch.zeros(i.edge_label.shape[0], dtype=torch.bool)
                if train:
                    edge_train_mask[...] = True
                else:
                    edge_val_mask[...] = True

                i.edge_train_mask = edge_train_mask
                i.edge_val_mask = edge_val_mask

            if hasattr(i, "sample_mapping"):
                sample_mapping = {}
                now = -1
                for j, k in enumerate(i.sample_mapping):
                    if k.item() != now:
                        now = k.item()
                        sample_mapping[k.item()] = j
                sample_mapping[now + 1] = i.sample_mapping.shape[0]
                i.samples = sample_mapping
                # print(sample_mapping)
            self.dataset_list.append(i)

    def len(self):
        return self.dataset_len

    def get(self, idx):
        t = idx % len(self.dataset_list)
        ret = self.dataset_list[t]
        # return ret
        if self.if_train:
            if hasattr(ret, "samples") and self.aug:
                sample_mapping = ret.samples
                n = len(sample_mapping) - 1
                pick = random.randint(0, max(0, n - 15)) + 15
                if n <= pick:
                    pick = n
                start = random.randint(0, n - pick)
                end = start + pick
                need_nodes = torch.zeros_like(ret.y, dtype=torch.bool)
                need_nodes[sample_mapping[start] : sample_mapping[end]] = True
                need_edges = (
                    need_nodes[ret.edge_index[0]] & need_nodes[ret.edge_index[1]]
                )
                ret2 = Data(x=ret.x, edge_index=ret.edge_index[:, need_edges])
                ret2.edge_train_mask = ret.edge_train_mask[need_edges]
                ret2.edge_val_mask = ret.edge_val_mask[need_edges]
                ret2.sample_mapping = ret.sample_mapping  # [need_nodes]
                ret2.y = ret.y  # [need_nodes]
                ret2.box_masks = ret.box_masks  # [need_nodes]
                ret2.boxes = ret.boxes  # [need_nodes]
                ret2.train_mask = ret.train_mask & need_nodes
                ret2.val_mask = ret.val_mask & need_nodes
                ret2.edge_label = ret.edge_label[need_edges]
                # print(ret2)
                return ret2
            else:
                return ret
        else:
            return ret

        return self.dataset_list[t]


class stage2MaskDataset(Dataset):
    def __init__(self, input_image_list, input_embed, target_masks, sample_mapping):
        self.input_image_list = input_image_list
        self.input_embed = input_embed
        self.target_mask = target_masks
        self.sample_mapping = sample_mapping

    def __len__(self):
        return self.target_mask.shape[0]

    def __getitem__(self, index):
        return (
            self.input_image_list[self.sample_mapping[index]],
            self.input_embed[index],
            self.target_mask[index],
        )


class stage2MaskDataModule(L.LightningDataModule):
    def __init__(
        self,
        input_image_list,
        input_embed,
        target_masks,
        sample_mapping,
        batch_size=2,
        train_val=[0.8, 0.2],
        seed=1013,
    ):
        super().__init__()
        self.dataset = stage2MaskDataset(
            input_image_list, input_embed, target_masks, sample_mapping
        )
        torch.manual_seed(seed)
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, train_val
        )
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size)


class stage2DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_list_train,
        dataset_list_val,
        dataset_len,
        ifaug=True,
    ):
        super().__init__()

        self.dataset_train = dataset_list_train
        self.dataset_val = dataset_list_val
        self.dataset_len = dataset_len
        self.ifaug = ifaug
        if ifaug:
            print("use node augmentation")

    def train_dataloader(self):
        # ds = stage2Dataset(self.dataset, self.dataset_len[0], self.train_val, self.seed)
        ds = stage2DatasetNew(
            self.dataset_train, self.dataset_len[0], True, aug=self.ifaug
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=1, collate_fn=utils.collect_graph, num_workers=8
        )

    def val_dataloader(self):
        # ds = stage2Dataset(self.dataset, self.dataset_len[1], self.train_val, self.seed)
        ds = stage2DatasetNew(self.dataset_val, self.dataset_len[1], False, aug=False)
        return torch.utils.data.DataLoader(
            ds, batch_size=1, collate_fn=utils.collect_graph, num_workers=1
        )


# used to separated different labels sets, only used for visualization
class CocoTraverse(torchvision.datasets.vision.VisionDataset):
    def __init__(
        self,
        image_directory_path: str,
        annotation_file_path: str,
        is_npy=True,
        filter_class=None,
        single_class=False,
        maxsize=800,
        add_classname=False,
        # map_class=None,
        require_mask=False,
        norm="zscore",
        filtermin=5,
    ):
        super().__init__(image_directory_path)
        self.coco = MyCOCO(annotation_file_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.is_npy = is_npy
        self.filter_class = filter_class
        self.single_class = single_class
        self.total_mapping_class = {}

        self.mapping_class = None
        # if map_class is not None:
        #     mc = {}
        #     for i, j in map_class.items():
        #         mc[int(i)] = j
        # else:
        #     mc = None
        # self.map_class = mc

        self.require_mask = require_mask
        # self.r_mapping_class = {}
        self.all_required = None
        self.maxsize = maxsize
        # for i in filter_class:
        #     if self.all_required is None:
        #         self.all_required = filter_class[i]
        #     else:
        #         self.all_required = np.intersect1d(self.all_required, filter_class[i])

        #     if self.single_class[i]:
        #         self.mapping_class[i] = {t: 0 for t in filter_class[i]}
        #         self.r_mapping_class[i] = {0: filter_class[i][0]}
        #         for j in filter_class[i]:
        #             self.total_mapping_class[j] = filter_class[i][0]
        #         if len(filter_class[i]) > 1:
        #             print(
        #                 "Warning: single class is set to True, but multiple classes are provided"
        #             )
        #             print(
        #                 "classes ",
        #                 filter_class[i],
        #                 " will be mapped to ",
        #                 filter_class[i][0],
        #             )
        #     else:
        #         self.mapping_class[i] = {
        #             t: idx for idx, t in enumerate(filter_class[i])
        #         }
        #         self.r_mapping_class[i] = {
        #             idx: t for idx, t in enumerate(filter_class[i])
        #         }
        #         for j in filter_class[i]:
        #             self.total_mapping_class[j] = j

        self.classes = pd.DataFrame(self.coco.dataset["categories"])
        self.classes = self.classes.set_index("id")
        self.add_classname = add_classname
        self.transform = utils.getConstantTransform()
        self.norm = norm
        self.filtermin = filtermin
        self._filterIds()

    def _load_image(self, id: int):
        if self.is_npy:
            path = self.coco.loadImgs(id)[0]["file_name"]
            path = os.path.join(self.root, path)
            # print(self.root, path)
            return np.load(path, allow_pickle=True)
        else:
            path = self.coco.loadImgs(id)[0]["file_name"]
            return Image.open(os.path.join(self.root, path)).convert("RGB")

    def processAnnotations(self, annotations, image):
        if len(annotations) == 0:
            return {
                "bboxes": [],
                "class_labels": [],
                "names": [],
                "labels": [],
                "item_id": [],
            }
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
        bbboxes = torch.stack(
            [torch.tensor(mask["bbox"]) for mask in annotations], dim=0
        )

        bt = torchvision.ops.box_convert(torch.Tensor(bbboxes), "xywh", "xyxy")
        boxes = BoundingBoxes(bt, format="xyxy", canvas_size=image.shape[1:])
        ret_dict = {"bboxes": boxes, "class_labels": labels}

        if "item_id" in annotations[0]:
            item_id = [sample["item_id"] for sample in annotations]
            ret_dict["item_id"] = item_id

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
            ret_dict["masks"] = Mask(masks)

        return ret_dict

        return {"bboxes": boxes, "class_labels": labels}

    def _load_target(self, id: int):
        if "zpos" in self.coco.loadImgs(id)[0]:
            pos = self.coco.loadImgs(id)[0]["zpos"]
        else:
            pos = 0
        t = self.coco.loadAnns(self.coco.getAnnIds(id))
        # print(t)
        ret = {}
        ret["pos"] = pos
        length = 0
        for i in self.filter_class:
            ret[i] = list(filter(lambda x: x["category_id"] in self.filter_class[i], t))
            length += len(ret[i])

        ret["num"] = length
        return ret

    def _getitem(self, index):
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        id = self.ids[index]
        image = self._load_image(id)
        if self.norm == "zscore":
            for i in range(3):
                image[i] = (image[i] - image[i].mean()) / image[i].std()
        elif self.norm == "hist":
            for i in range(3):
                image[i] = exposure.equalize_hist(image[i])
        target = self._load_target(id)

        return image, target

    def _filterIds(self):
        needids = []
        for i in range(len(self.ids)):
            annotation = self._load_target(self.ids[i])
            if annotation["num"] > self.filtermin:
                needids.append(i)
                # if i == 6:
                #     print(i, len(annotation))
            # if len(self.coco.loadAnns(self.coco.getAnnIds(i))) > 0:
            #     need.append(i)
        self.needids = needids

    def __len__(self):
        return len(self.needids)

    def normalize_target(self, target):
        if len(target["bboxes"]) == 0:
            return target
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

        target["iscrowd"] = torch.zeros(
            (len(target["class_labels"])), dtype=torch.int64
        )

        if self.add_classname:
            target["names"] = []
            for i in target["class_labels"]:
                target["names"].append(self.classes.loc[i.item()]["name"])

        return target

    def __getitem__(self, index):
        if self.needids is not None:
            idx = self.needids[index]

        image, annotation = self._getitem(idx)

        image = torch.tensor(image)
        for i in self.filter_class.keys():
            annotation[i] = self.processAnnotations(annotation[i], image)

        c, h, w = image.shape

        if h > self.maxsize or w > self.maxsize:
            hq = h if h < self.maxsize else self.maxsize
            hq = hq / h
            wq = w if w < self.maxsize else self.maxsize
            wq = wq / w
            r = min(hq, wq)
            temp = transforms.Compose(
                [
                    transforms.Resize((int(r * h), int(r * w))),
                    transforms.SanitizeBoundingBoxes(),
                ]
            )
            for i in self.filter_class.keys():
                if len(annotation[i]["bboxes"]) >= 1:
                    image, annotation[i] = temp(image, annotation[i])

        padtransform = transforms.Pad(
            (0, 0, self.maxsize - int(r * w), self.maxsize - int(r * h)), fill=0
        )
        Flag = 0
        oldimage = image
        for i in self.filter_class.keys():
            Flag += 1
            if len(annotation[i]["bboxes"]) >= 1:
                image, annotation[i] = padtransform(oldimage, annotation[i])
            # else:
            #     _, annotation[i] = padtransform(oldimage, annotation[i])

        mask = torch.zeros((self.maxsize, self.maxsize), dtype=torch.long)
        mask[:h, :w] = 1

        for i in self.filter_class.keys():
            annotation[i] = self.normalize_target(annotation[i])

        annotation["orig_size"] = torch.tensor((h, w))
        annotation["size"] = (int(r * h), int(r * w))

        ret = {
            "pixel_values": image,
            "pixel_mask": mask,
            "labels": annotation,
        }

        # print(ret)

        return ret


# main class
class CocoDetection(torchvision.datasets.vision.VisionDataset):

    def __init__(
        self,
        image_directory_path: str,
        annotation_file_path: str,
        is_npy=True,
        require_mask=False,
        filter_class=None,
        single_class=False,
        map_class=None,
        transform=None,
        add_classname=False,
        maxsize=800,
        return_single=None,
        mark="",
        norm="none",
        filtermin=5,
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
        if map_class is not None:
            mc = {}
            for i, j in map_class.items():
                mc[int(i)] = j
        else:
            mc = None
        self.map_class = mc
        # if self.filter_class is not None:
        #     if self.single_class:
        #         self.mapping_class = {i: 0 for i in filter_class}
        #     else:
        #         self.mapping_class = {i: idx for idx, i in enumerate(filter_class)}
        # else:
        #     self.mapping_class = None

        self.transform = transform
        self.classes = pd.DataFrame(self.coco.dataset["categories"])
        self.classes = self.classes.set_index("id")
        self.add_classname = add_classname
        self.maxsize = maxsize
        self.norm = norm

        self.return_single = return_single
        self.filtermin = filtermin
        self._filterIds()
        # self.zpos = []

    def _load_image(self, id: int):
        if self.is_npy:
            path = self.coco.loadImgs(id)[0]["file_name"]
            # print(self.root, path)
            path = os.path.join(self.root, path)
            return np.load(path, allow_pickle=True)
        else:
            path = self.coco.loadImgs(id)[0]["file_name"]
            return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        if "zpos" in self.coco.loadImgs(id)[0]:
            pos = self.coco.loadImgs(id)[0]["zpos"]
        else:
            pos = 0
        t = self.coco.loadAnns(self.coco.getAnnIds(id))

        if self.filter_class is not None:
            t = list(filter(lambda x: x["category_id"] in self.filter_class, t))

        if len(t) < self.filtermin:
            return [], 0
        return t, pos

    def processAnnotations(self, annotations, image):
        # print(image.shape)
        # if self.mapping_class is not None:
        #     labels = torch.tensor(
        #         [self.mapping_class[sample["category_id"]] for sample in annotations],
        #         dtype=torch.long,
        #     )
        # else:
        labels = torch.tensor(
            [sample["category_id"] for sample in annotations], dtype=torch.long
        )

        if self.add_classname:
            names = []
            for i in labels:
                names.append(self.classes.loc[i.item()]["name"])

        if self.single_class:
            assert self.map_class is None
            labels = torch.zeros_like(labels)

        if self.map_class is not None:
            labels = torch.tensor(
                [self.map_class[sample["category_id"]] for sample in annotations],
                dtype=torch.long,
            )

        # labels = [str(sample["category_id"]) for sample in annotations]
        bbboxes = torch.stack(
            [torch.tensor(mask["bbox"]) for mask in annotations], dim=0
        )

        bt = torchvision.ops.box_convert(torch.Tensor(bbboxes), "xywh", "xyxy")
        boxes = BoundingBoxes(bt, format="xyxy", canvas_size=image.shape[1:])

        retdict = {"bboxes": boxes, "class_labels": labels}

        if "item_id" in annotations[0]:
            item_id = [sample["item_id"] for sample in annotations]
            retdict["item_id"] = item_id

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
            retdict["masks"] = Mask(masks)
        #     return {"masks": Mask(masks), "bboxes": boxes, "class_labels": labels}
        # else:
        #     return {"bboxes": boxes, "class_labels": labels}
        if self.add_classname:
            retdict["names"] = names
        return retdict

    def __len__(self):
        return len(self.needids)

    def _filterIds(self):
        needids = []
        zposes = []
        itemids = {}
        for i in range(len(self.ids)):
            annotation, zpos = self._load_target(self.ids[i])
            if len(annotation) >= self.filtermin:
                needids.append(i)
                zposes.append(zpos)
            # if len(self.coco.loadAnns(self.coco.getAnnIds(i))) > 0:
            #     need.append(i)
        self.needids = needids
        self.zpos = zposes
        # print(needids)

    def _getitem(self, index):
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        id = self.ids[index]
        image = self._load_image(id)
        if self.norm == "zscore":
            for i in range(3):
                image[i] = (image[i] - image[i].mean()) / image[i].std()
        elif self.norm == "hist":
            for i in range(3):
                image[i] = exposure.equalize_hist(image[i])
        target, _ = self._load_target(id)

        return image, target

    def __getitem__(self, idx):
        # idx1 = idx
        if self.needids is not None:
            zpos = self.zpos[idx]
            idx = self.needids[idx]

        # used to return a single image, used for debugging
        if self.return_single is not None:
            idx = self.return_single
        # image, annotation = super().__getitem__(idx)
        image, annotation = self._getitem(idx)

        # print(idx1, idx, len(annotation))
        if len(annotation) == 0:
            raise ValueError
        image = torch.tensor(image)
        target = self.processAnnotations(annotation, image)

        ori_class_labels = target["class_labels"].clone()
        target["class_labels"] = torch.tensor(
            range(len(target["class_labels"])), dtype=torch.long
        )

        if self.transform is not None:
            image, target = self.transform(image, target)

        c, h, w = image.shape
        target["orig_size"] = torch.tensor((h, w))
        if h > self.maxsize or w > self.maxsize:
            hq = h if h < self.maxsize else self.maxsize
            hq = hq / h
            wq = w if w < self.maxsize else self.maxsize
            wq = wq / w
            r = min(hq, wq)
            temp = transforms.Compose(
                [
                    transforms.Resize((int(r * h), int(r * w))),
                    transforms.SanitizeBoundingBoxes(),
                ]
            )
            image, target = temp(image, target)
            h = int(r * h)
            w = int(r * w)

        target["size"] = torch.tensor((h, w))

        mask = torch.zeros((self.maxsize, self.maxsize), dtype=torch.long)
        mask[:h, :w] = 1

        padtransform = transforms.Pad(
            (0, 0, self.maxsize - w, self.maxsize - h), fill=0
        )
        image, target = padtransform(image, target)

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

        target["pos"] = zpos
        # print(target["class_labels"])
        for i in ["names", "item_id"]:
            if i in target:
                target[i] = [target[i][j.item()] for j in target["class_labels"]]
        target["class_labels"] = ori_class_labels[target["class_labels"]]

        ret = {
            "pixel_values": image,
            "pixel_mask": mask,
            "labels": target,
        }
        if self.mark is not None:
            ret["mark"] = self.mark
        return ret


class CocoDetection2(CocoDetection):
    def __init__(self, num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("using batch size ", num)
        self.num = num
        self.seed = 42

    def __len__(self):
        return super().__len__() - self.num + 1

    def __getitem__(self, idx):
        self.seed = self.seed + 1
        res = []
        for i in range(self.num):
            torch.manual_seed(self.seed)
            res.append(super().__getitem__(idx + i))
        return utils.stackBatch(res)


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


class AdditionalInputLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, in_dim * 10)
        self.layer2 = nn.Linear(in_dim * 10, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x


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
            feature_dim + additional_output_dim,
            additional_input_dim,
            output_dim,
            layer_type=layer_type,
            dropout=dropout,
        )
        t = torch.ones((output_dim))
        t[-1] = 0.1
        self.cri = nn.CrossEntropyLoss(weight=t)
        self.output_dim = output_dim
        self.mask_head = UNet(
            embed_dim=feature_dim + additional_output_dim, sigmoid=False
        )

        self.scheduler_step = scheduler_step

    def forward(
        self,
        x=None,
        edge_index=None,
        pixel_values=None,
        pixel_mask=None,
        labels=None,
        mark=None,
        stage_2_embeds=None,
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
            return self.mask_head(
                pixel_values,
                stage_2_embeds,
            )

        else:
            raise NotImplementedError

    def loss_boxes(self, source_boxes, targets, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        loss_bbox = nn.functional.mse_loss(source_boxes, targets)

        # losses = {}
        # losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # loss_giou = 1 - torch.diag(
        #     generalized_box_iou(
        #         center_to_corners_format(source_boxes),
        #         center_to_corners_format(targets),
        #     )
        # )
        # print(source_boxes[0], targets[0])
        # print(
        #     loss_bbox,
        #     loss_giou,
        # )
        # loss_bbox = loss_bbox.mean()
        # loss_giou = loss_giou.mean()
        # losses["loss_giou"] = loss_giou.sum() / num_boxes
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
                    loss_dict = outputs[mark].loss_dict
                else:
                    for i in loss_dict:
                        loss_dict[i] += outputs[mark].loss_dict[i]
            else:
                loss = outputs.loss
                loss_dict = outputs.loss_dict
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
        self, x, edge, mask, y, box=None, box_mask=None, edge_label=None, edge_mask=None
    ):

        x = x.to(self.device)
        edge = edge.to(self.device)
        mask = mask.to(self.device)
        y = y.to(self.device)

        ret_dict = self(x=x, edge_index=edge)
        # print(ret_dict["predict"][mask].shape, y[mask].shape)
        # print(torch.max(y[mask]))
        loss = self.cri(ret_dict["predict"][mask], y[mask])
        loss_dict = {
            "loss": loss,
            "auroc": self.auroc(ret_dict["predict"][mask].detach(), y[mask]),
        }

        if box is not None:
            box_mask = box_mask.to(self.device)
            box = box.to(self.device)
            box_mask = box_mask & mask
            loss_boxes = self.loss_boxes(
                ret_dict["box"][box_mask], box[box_mask], box_mask.sum()
            )
            loss += loss_boxes
            loss_dict["loss_boxes"] = loss_boxes
        else:
            loss_boxes = 0

        if edge_label is not None:
            edge_label = edge_label.to(self.device)
            if edge_mask is None:
                edge_mask = torch.ones_like(edge_label, dtype=torch.bool)
            loss_edge = F.binary_cross_entropy(
                ret_dict["edge"].squeeze()[edge_mask], edge_label[edge_mask].float()
            )
            loss += loss_edge
            loss_dict["loss_edge"] = loss_edge

        return loss, loss_dict

    # def focal_loss(self, inputs, targets, alpha: float = 0.75, gamma: float = 2):

    #     ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    #     # add modulating factor
    #     p_t = inputs.detach() * targets + (1 - inputs.detach()) * (1 - targets)
    #     loss = ce_loss * ((1 - p_t) ** gamma)

    #     if alpha >= 0:
    #         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    #         loss = alpha_t * loss

    #     return loss.sum()

    def common_stage_mask(self, pixel_values, stage_2_embeds, mask, cal_auroc=False):
        pixel_values = pixel_values.to(self.device).float()
        stage_2_embeds = stage_2_embeds.to(self.device).float()
        mask = mask.to(self.device).float()
        outputs = self(pixel_values=pixel_values, stage_2_embeds=stage_2_embeds)
        # print(mask.shape, outputs.shape)
        # print(outputs[0][:5])
        # print(mask[0][:5])
        loss = sigmoid_focal_loss(outputs, mask.float())
        if cal_auroc:
            auroc = self.mask_auroc(
                outputs.detach().view(-1).cpu(), mask.detach().view(-1).cpu()
            )
            return loss, {"loss": loss, "auroc": auroc}
        # loss = F.binary_cross_entropy(outputs, mask.float())
        return loss, {"loss": loss}

    def on_validation_epoch_end(self):
        # loss = torch.stack(self.val_step_outputs).mean()
        losses = np.mean([i["loss"] for i in self.val_step_outputs])
        if "stage mask" in self.stage:
            auroc = np.mean([i["auroc"] for i in self.val_step_outputs])
            self.log("total_validate_auroc", auroc)
        if "stage 1 + 2" in self.stage:
            auroc = np.mean([i["auroc"] for i in self.val_step_outputs])
            self.log("total_validate_auroc", auroc, prog_bar=True)
            loss_boxes = np.mean(
                [i["loss_boxes"] for i in self.val_step_outputs if "loss_boxes" in i]
            )
            self.log("total_validate_loss_boxes", loss_boxes, prog_bar=True)
        # loss = np.mean(self.val_step_outputs)
        self.log("total_validate_loss", losses, prog_bar=True)
        self.val_step_outputs.clear()

    def training_step(self, batch, batch_idx=0, loader_idx=0):
        # with open("./temp.pkl", "wb") as f:
        #     pickle.dump(batch, f)
        # print(batch)
        if "stage 1 + 2" in self.stage:
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
            edge_label = data2.edge_label if hasattr(batch, "edge_label") else None
            # edge_label = None
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

            pixel_values, stage_2_embeds, pixel_mask = batch
            # print(pixel_mask)
            loss, loss_dict = self.common_stage_mask(
                pixel_values, stage_2_embeds, pixel_mask
            )

        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx=0, loader_idx=0):
        # print(batch)

        if "stage 1 + 2" in self.stage:
            loss, loss_dict, output = self._common_step_stage1(batch[0], 0, None, True)
            retdict = utils.process(output, batch[0]["labels"])
            data2 = utils.convertStage2Dataset(retdict)
            self.stage = "stage 2"
            x = data2.x
            y = data2.y
            edge_index = data2.edge_index
            mask = torch.ones_like(y, dtype=torch.bool)
            boxes = data2.boxes if hasattr(data2, "boxes") else None
            edge_label = data2.edge_label if hasattr(batch, "edge_label") else None
            # edge_label = None
            box_masks = data2.box_masks if hasattr(data2, "box_masks") else None
            loss2, loss_dict2 = self.common_step_stage2(
                x, edge_index, mask, y, boxes, box_masks, edge_label
            )
            self.stage = "stage 1 + 2"
            loss = loss + loss2
            loss_dict = loss_dict2
        elif "stage 1" in self.stage:
            loss, loss_dict = self.common_step_stage1(batch)
        elif "stage 2" in self.stage:
            mask = batch.val_mask
            y = batch.y
            boxes = batch.boxes if hasattr(batch, "boxes") else None
            box_masks = batch.box_masks if hasattr(batch, "box_masks") else None
            edge_label = batch.edge_label if hasattr(batch, "edge_label") else None
            edge_mask = batch.val_edge_mask if hasattr(batch, "val_edge_mask") else None
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
            pixel_values, stage_2_embeds, pixel_mask = batch
            loss, loss_dict = self.common_stage_mask(
                pixel_values, stage_2_embeds, pixel_mask, True
            )
        res = {}
        res["loss"] = loss.detach().cpu()
        for k, v in loss_dict.items():
            res[k] = v.detach().cpu()
        self.val_step_outputs.append(res)
        self.log("validation_loss", res["loss"])
        for k, v in loss_dict.items():
            self.log("validate_" + k, v.item(), prog_bar=False)

        return loss

    def configure_optimizers(self):
        optim = None
        if "stage 1" in self.stage:
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

            # return {
            #     "optimizer": optim,
            #     "lr_scheduler": {
            #         "scheduler": torch.optim.lr_scheduler.StepLR(
            #             optim, step_size=self.scheduler_step, gamma=0.5
            #         ),
            #         "interval": "epoch",
            #     },
            # }
        else:
            return optim

        return torch.optim.Adam(
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

        pixel_values = batch[0]["pixel_values"]
        pixel_mask = batch[0]["pixel_mask"]
        labels = [
            {k: v.to(self.device) for k, v in t.items()} for t in batch[0]["labels"]
        ]

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
    def __init__(
        self, trainsets, valsets, train_batch_size, val_batch_size, stack_batch=True
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.trainset = trainsets
        self.valset = valsets
        if stack_batch:
            self.stack_batch = utils.stackBatch
        else:
            self.stack_batch = utils.identicalMapping

    def train_dataloader(self):

        if not isinstance(self.trainset, dict):
            return torch.utils.data.DataLoader(
                dataset=self.trainset,
                collate_fn=self.stack_batch,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=4,
            )

        train_sets = []
        for i in self.trainset:
            train_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.trainset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    num_workers=4,
                )
            )
        return CombinedLoader(
            train_sets,
            mode="max_size_cycle",  # Ensures cycling through both dataloaders
        )

    def val_dataloader(self):

        if not isinstance(self.valset, dict):
            return torch.utils.data.DataLoader(
                dataset=self.valset,
                collate_fn=self.stack_batch,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=4,
            )

        val_sets = []
        for i in self.valset:
            val_sets.append(
                torch.utils.data.DataLoader(
                    dataset=self.valset[i],
                    collate_fn=self.stack_batch,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    num_workers=4,
                )
            )
        return CombinedLoader(
            val_sets, mode="max_size_cycle"  # Ensures cycling through both dataloaders
        )

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
    ):

        super().__init__()
        self.output_classes = output_classes
        self.additional_input_dim = additional_input_dim
        self.additional_input_layer = AdditionalInputLayer(additional_input_dim, 16)
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

    def forward(self, inputs, edge_index):

        additional_input, model_feature = (
            inputs[:, : self.additional_input_dim],
            inputs[:, self.additional_input_dim :],
        )

        additional_feature = self.additional_input_layer(additional_input)
        if self.dropout:
            model_feature = F.dropout(model_feature, p=0.2)
        x = torch.cat([model_feature, additional_feature], dim=1)

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # if self.dropout:
        #     x = F.dropout(x, p=0.2)
        x = self.conv2(x, edge_index)

        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=0.2)
        x = self.conv3(x, edge_index)

        predict = self.cls_head(x)
        # predict[:, : self.output_classes - 1] += inputs[:, 5 : self.output_classes + 4]

        box = self.box_head(x) + inputs[:, 1:5]

        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=1)
        edge = self.edge_head(edge_embeddings)

        return {"predict": predict, "box": box, "edge": edge, "embeddings": x}


class UNetWithAdditionalInput(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, additional_input_dim=10):
        super(UNetWithAdditionalInput, self).__init__()

        # Contracting path (Encoder)
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)

        # Process additional input
        self.additional_fc = nn.Sequential(
            nn.Linear(
                additional_input_dim, 1024
            ),  # Map additional input to the same dimension as bottleneck
            nn.ReLU(inplace=True),
        )

        # Expanding path (Decoder)
        self.upconv4 = self.upconv(
            2048, 512
        )  # Bottleneck + Additional input (1024 + 1024 = 2048)
        self.dec4 = self.double_conv(1024, 512)

        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.double_conv(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.double_conv(256, 128)

        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.double_conv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """Double convolution layers with ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv(self, in_channels, out_channels):
        """Upsample the feature map using transpose convolution."""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def crop_and_concat(self, upsampled, bypass):
        """Crop the bypass connection to match the size of the upsampled feature map."""
        _, _, h, w = upsampled.size()
        bypass = F.interpolate(bypass, size=(h, w), mode="bilinear", align_corners=True)
        return torch.cat((upsampled, bypass), dim=1)

    def forward(self, x, additional_input):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Process additional input
        additional_input_processed = self.additional_fc(additional_input)
        additional_input_processed = additional_input_processed.unsqueeze(-1).unsqueeze(
            -1
        )  # Reshape to (batch, 1024, 1, 1)
        additional_input_processed = additional_input_processed.expand(
            -1, -1, bottleneck.size(2), bottleneck.size(3)
        )  # Match spatial dims

        # Concatenate bottleneck and additional input
        bottleneck = torch.cat((bottleneck, additional_input_processed), dim=1)

        # Decoder
        up4 = self.upconv4(bottleneck)
        concat4 = self.crop_and_concat(up4, enc4)
        dec4 = self.dec4(concat4)

        up3 = self.upconv3(dec4)
        concat3 = self.crop_and_concat(up3, enc3)
        dec3 = self.dec3(concat3)

        up2 = self.upconv2(dec3)
        concat2 = self.crop_and_concat(up2, enc2)
        dec2 = self.dec2(concat2)

        up1 = self.upconv1(dec2)
        concat1 = self.crop_and_concat(up1, enc1)
        dec1 = self.dec1(concat1)

        # Output layer
        output = self.out_conv(dec1)
        output = output.squeeze(1)  # Remove channel dimension
        output = torch.sigmoid(output)  # Sigmoid activation for binary classification
        return output


class SelfAttention(nn.Module):
    def __init__(self, channels, size, pos_embed=16):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels + pos_embed, 4, batch_first=True)
        self.pos = nn.Parameter(torch.randn(1, size * size, pos_embed))
        self.l1 = nn.Linear(channels + pos_embed, channels)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        x_ln = torch.cat([x_ln, self.pos.repeat(x.shape[0], 1, 1)], dim=-1)

        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = self.l1(attention_value)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )


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
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, pool_kernal=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_kernal),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, size, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(size=size, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=1, embed_dim=272, sigmoid=True):
        super().__init__()

        # 64 * 800 * 800
        self.inc = DoubleConv(c_in, 32)

        # 128 * 200 * 200
        self.down1 = Down(32, 64, embed_dim, 4)

        # 256 * 50 * 50
        self.down2 = Down(64, 128, embed_dim, 4)
        # effective patch size 16 * 16
        self.sa1 = SelfAttention(128, 50)

        # 256 * 25 * 25
        self.down3 = Down(128, 256, embed_dim)
        # effective patch size 32 * 32
        self.sa2 = SelfAttention(256, 25)

        # 256 * 12 * 12
        self.down4 = Down(256, 256, embed_dim)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # 256 * 25 * 25
        self.up1 = Up(512, 128, (25, 25), embed_dim)
        self.sa4 = SelfAttention(128, 25)

        # 128*50*50
        self.up2 = Up(256, 64, (50, 50), embed_dim)
        self.sa5 = SelfAttention(64, 50)

        # 64 * 200 * 200
        self.up3 = Up(128, 32, (200, 200), embed_dim)

        # 32 * 800 * 800
        self.up4 = Up(64, 64, (800, 800), embed_dim)

        # 1 * 800 * 800
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.sigmoid = sigmoid

    def forward(self, x, t):

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x3 = self.sa1(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa2(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa3(x5)

        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x4, t)
        x = self.sa4(x)
        x = self.up2(x, x3, t)
        x = self.sa5(x)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        output = self.outc(x).squeeze(1)
        if self.sigmoid:
            output = torch.sigmoid(output)
        return output
