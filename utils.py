import json
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
from tqdm import tqdm
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
            transforms.RandomIoUCrop(min_scale=0.8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.SanitizeBoundingBoxes(min_size=5),
        ]
    )
    return allt


def getSimpleTransform():

    allt = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True),
            # transforms.RandomResize(600, 1000),
            # transforms.Lambda(lambda x:torch.clamp(x, min=-4.0, max=4.0)),
            # transforms.RandomIoUCrop(),
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


def drawannotation(image, target, box=True, mask=True, font_size=30):
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    if isinstance(image, torch.Tensor):
        # image = image.numpy()
        # if np.max(image) < 10:
        image -= torch.min(image)
        image /= torch.max(image)
        image = torch.tensor(image * 255).type(torch.uint8)

    if "masks" in target and mask:
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
    if box:
        annotated_tensor = draw_bounding_boxes(
            image=annotated_tensor,
            boxes=target["bboxes"],
            labels=target["names"] if "names" in target else target["labels"],
            font_size=font_size,
            width=5,
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


def identicalMapping(batch):
    return batch


def collect_graph(batch):
    return batch[0]


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
        if isinstance(classids, torch.Tensor):
            ids = torch.where(classids == i)[0]
        else:
            ids = np.where(classids == i)[0]
            ids = torch.tensor(ids)
        if ids.dim() < 1:
            ids = ids.unsqueeze(0)
        # if len(keep) >= 1:
        # allkeep.append(torch.tensor([]))
        allkeep.append(ids[keep])
    # print(allkeep)
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
def postSegmentationTreatment(
    outputs, threshold, target_sizes, mask_threshold=0.5, style="without None"
):
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

    if style == "without None":
        prob = nn.functional.sigmoid(out_logits)
        scores, labels = prob.max(-1)
        is_nones = torch.zeros_like(scores)
    else:
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., 1:].max(-1)
        is_nones = prob[..., 0]

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
        for s, l, b, mask, size, is_none in zip(
            scores, labels, boxes, masks, target_sizes, is_nones
        ):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            mask = mask[s > threshold]
            is_none = is_none[s > threshold]
            print(mask.shape, size)
            if np.any(mask.shape[1:2] != size):
                mask = nn.functional.interpolate(
                    mask[:, None], size=to_tuple(size), mode="bilinear"
                )
                mask = mask > mask_threshold

            results.append(
                {
                    "scores": score,
                    "labels": label,
                    "boxes": box,
                    "masks": mask,
                    "is_none": is_none,
                }
            )
    else:
        for s, l, b, size, is_nones in zip(
            scores, labels, boxes, target_sizes, is_nones
        ):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            is_none = is_nones[s > threshold]
            results.append(
                {"scores": score, "labels": label, "boxes": box, "is_none": is_none}
            )
    return results


def toTarget(result, size):
    if "masks" in result:
        return {
            "labels": result["labels"],
            "bboxes": BoundingBoxes(result["boxes"], format="xyxy", canvas_size=size),
            "masks": Mask(result["masks"].squeeze(1)),
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

    if (
        "renew_position_embeddings" in configs["model"]
        and configs["model"]["renew_position_embeddings"]
    ):
        print("renew position embeddings")
        state_dict = model.state_dict()
        for i in state_dict:
            if "query_position_embeddings" in i:
                state_dict[i] = torch.randn(state_dict[i].shape)
        model.load_state_dict(state_dict)

    if checkpoint is not None and len(checkpoint) > 0:
        if checkpoint.endswith(".ckpt"):
            ckpt = torch.load(checkpoint)["state_dict"]
        else:
            ckpt = torch.load(checkpoint)
        for i, j in model.named_parameters():
            if i in ckpt and j.shape != ckpt[i].shape:
                if "query_position_embeddings" not in i:
                    del ckpt[i]
                else:
                    t = j.clone()
                    t[: min(t.shape[0], ckpt[i].shape[0])] = ckpt[i][
                        : min(t.shape[0], ckpt[i].shape[0])
                    ]
                    # t = torch.randn_like(j)
                    ckpt[i] = t
        need_del = []
        p = [j for j, _ in model.named_parameters()]
        for i in ckpt:
            if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
                need_del.append(i)
        for i in need_del:
            del ckpt[i]
        model.load_state_dict(ckpt, strict=False)

    return model


def getModel(configs):

    if "checkpoint" not in configs["model"]:
        configs["model"]["checkpoint"] = None

    if isinstance(configs["data"]["require_mask"], dict):

        models = {}

        for i in configs["model"]["args"]:
            models[i] = buildModel(
                configs, configs["model"]["args"][i], configs["model"]["checkpoint"]
            )
    else:
        models = buildModel(
            configs, configs["model"]["args"], configs["model"]["checkpoint"]
        )

    if "lr_backbone" not in configs["training"]:
        configs["training"]["lr_backbone"] = None

    if "lr_detr" not in configs["training"]:
        configs["training"]["lr_detr"] = None

    if "dropout" not in configs["model"]:
        configs["model"]["dropout"] = False

    if "additional_input_dim" not in configs["model"]:
        configs["model"]["additional_input_dim"] = 20

    if "mask_model" not in configs["model"]:
        configs["model"]["mask_model"] = "Unet6"

    if "scheduler_step" not in configs["training"]:
        configs["training"]["scheduler_step"] = -1

    if "warmup_epoches" not in configs["training"]:
        configs["training"]["warmup_epoches"] = 0

    model = modules.DetrModel(
        configs["model"]["stage"],
        models,
        lr=configs["training"]["lr"],
        lr_backbone=configs["training"]["lr_backbone"],
        lr_detr=configs["training"]["lr_detr"],
        weight_decay=configs["training"]["weight_decay"],
        feature_dim=configs["model"]["feature_dim"],
        additional_input_dim=configs["model"]["additional_input_dim"],
        output_dim=configs["model"]["output_dim"],
        layer_type=configs["model"]["layer_type"],
        dropout=configs["model"]["dropout"],
        scheduler_step=configs["training"]["scheduler_step"],
        warmup_epoches=configs["training"]["warmup_epoches"],
        mask_model=configs["model"]["mask_model"],
    )
    if "load" in configs["model"] and configs["model"]["load"] is not None:
        t = torch.load(configs["model"]["load"], map_location="cpu")
        ckpt = t["state_dict"]
        need_del = []
        p = list(model.state_dict().keys())  # [j for j, _ in model.named_parameters()]
        # print("model parameters ", p)
        for i in ckpt:
            if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
                need_del.append(i)
        for i in need_del:
            del ckpt[i]
        print("incompatible parameters", need_del)
        model.load_state_dict(ckpt, strict=False)
        print("finish loading parameters")

    return model


from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrHungarianMatcher,
)

matcher = DeformableDetrHungarianMatcher(1.0, 5.0, 2.0)
matcher


def processSingle(model, label, data, target_size, thres, has_none, empty=5):
    ret = []
    boxeses = []
    box_masks = []
    l = []
    item_ids = []
    masks = []
    output = model(
        pixel_values=data["pixel_values"].unsqueeze(0).float(),
        pixel_mask=data["pixel_mask"].unsqueeze(0).float(),
    )
    # for idx, i in enumerate(model_names):
    #     # print(i)
    #     if i!="none":
    #         _output = output[i]
    #     else:
    #         _output = output
    logits = output["logits"].squeeze(0)
    pred_boxes = output["pred_boxes"].squeeze(0)
    if has_none:
        # print(logits, logits.shape)
        prob = torch.softmax(logits, -1)
        prob = prob[:, :-1]
    else:
        prob = torch.sigmoid(logits)

    if isinstance(thres, float):
        v, pos = torch.max(prob, dim=1)
        reserve = v > thres
    else:
        reserve = reserve > thres
        reserve = reserve.any(dim=1)

    if label["class_labels"] is not None:
        o = {}
        o["pred_boxes"] = pred_boxes.unsqueeze(0)
        o["logits"] = logits.unsqueeze(0)
        match_res = matcher(o, [label])
        match_res = match_res[0]
        r = torch.zeros_like(reserve, dtype=torch.bool)
        r[match_res[0]] = True
        reserve2 = reserve | r
        print("check whether reserve ", sum(reserve), sum(reserve2))
        reserve = reserve2
    # else:
    #     reserve = v > thres

    # v, pos = torch.max(prob, dim=1)
    logits = logits[reserve]
    pred_boxes = pred_boxes[reserve]
    embed = output["last_hidden_state"].squeeze(0)
    embed = embed[reserve]

    pos = torch.zeros((pred_boxes.shape[0], 5))
    pos[:, 0] = label["pos"]
    if "zposmax" in label:
        pos[:, 0] /= label["zposmax"]
    else:
        pos[:, 0] /= 500
    # print(pred_boxes)
    pos[:, 1] = pred_boxes[:, 0]  # * (label["size"][0] / target_size[0])
    pos[:, 2] = pred_boxes[:, 1]  # * (label["size"][1] / target_size[1])
    pos[:, 3] = pred_boxes[:, 2]  # * (label["size"][0] / target_size[0])
    pos[:, 4] = pred_boxes[:, 3]  # * (label["size"][1] / target_size[1])
    input = torch.concat([pos, logits, embed], dim=1)
    ret.append(input)
    targets = torch.zeros((pred_boxes.shape[0]), dtype=torch.long)
    targets.fill_(empty)
    box_mask = torch.zeros((pred_boxes.shape[0]), dtype=torch.bool)
    boxes = torch.zeros((pred_boxes.shape[0], 4))
    item_id = ["" for i in range(pred_boxes.shape[0])]

    if label["class_labels"] is not None:
        if len(label["class_labels"]) > 0:
            o = {}
            o["pred_boxes"] = pred_boxes.unsqueeze(0)
            o["logits"] = logits.unsqueeze(0)
            # print(o, label[i])
            match_res = matcher(o, [label])
            match_res = match_res[0]
            target = label["class_labels"]
            target = target[match_res[1]]
            # print(targets, target)
            target_boxes = label["boxes"]
            target_boxes = target_boxes[match_res[1]]

            if "item_id" in label:
                for s, t in zip(match_res[0], match_res[1]):
                    item_id[s] = label["item_id"][t]

            boxes[match_res[0]] = target_boxes
            box_mask[match_res[0]] = True
            print(box_mask.shape, boxes.shape)
            targets[match_res[0]] = target

            if "masks" in label:
                masks.append(label["masks"][match_res[1]])

    l.append(targets)
    box_masks.append(box_mask)
    boxeses.append(boxes)
    item_ids.append(item_id)

    return {
        "feature": ret,
        "label": l,
        "box_mask": box_masks,
        "boxes": boxeses,
        "item_id": item_ids,
        "masks": masks,
    }
    return ret, l, box_masks, boxeses


def buildStage2(
    model,
    dataset,
    target_size,
    thres,
    # model_names=["other", "ribo"],
    has_none=False,
    empty=5,
):
    ret = []
    l = []
    box_masks = []
    masks = []
    boxes = []
    item_ids = []
    images = []
    pixel_masks = []
    sample_mapping = {}
    ret_dict = {
        "feature": ret,
        "label": l,
        "masks": masks,
        "box_mask": box_masks,
        "boxes": boxes,
        "item_id": item_ids,
        "images": images,
        "pixel_masks": pixel_masks,
        "sample_mapping": sample_mapping,
    }
    cnts = 0
    for i in range(len(dataset)):
        print("slice", i)
        data = dataset[i]
        label = data["labels"]
        with torch.no_grad():
            _ret_dict = processSingle(
                model, label, data, target_size, thres, has_none, empty
            )

        for j in _ret_dict:
            ret_dict[j].extend(_ret_dict[j])
        ret_dict["images"].append(data["pixel_values"])
        ret_dict["pixel_masks"].append(data["pixel_mask"])
        # print(_ret_dict["item_id"])
        for i in _ret_dict["item_id"]:
            for j in range(len(i)):
                sample_mapping[cnts] = len(ret_dict["images"]) - 1
                cnts += 1

        print("obj_cnts:", cnts)

    ret_dict["feature"] = torch.cat(ret_dict["feature"], dim=0)
    ret_dict["label"] = torch.cat(ret_dict["label"], dim=0)
    ret_dict["box_mask"] = torch.cat(ret_dict["box_mask"], dim=0)
    ret_dict["boxes"] = torch.cat(ret_dict["boxes"], dim=0)
    if len(ret_dict["masks"]) > 0:
        ret_dict["masks"] = torch.cat(ret_dict["masks"], dim=0)
    # else:
    #     ret_dict["masks"] = None
    item_ids = []
    for i in ret_dict["item_id"]:
        item_ids.extend(i)
    ret_dict["item_id"] = item_ids
    return ret_dict
    # return ret, l, box_masks, boxes
    # return ret, l


def process(outputs, labels, empty=5, need_mask=False):
    ret = []
    boxeses = []
    box_masks = []
    l = []
    item_ids = []
    mask = []
    logits = outputs["logits"]
    device = logits.device
    slice_num, num_obj, _ = logits.shape
    pred_boxes = outputs["pred_boxes"]

    match_res = matcher(outputs, labels)

    cnts = 0
    for i in range(slice_num):
        label = labels[i]
        pos = torch.zeros((num_obj, 5)).to(device)
        pos[:, 0] = label["pos"]
        if "zposmax" in label:
            pos[:, 0] /= label["zposmax"]
        else:
            pos[:, 0] /= 500
        pos[:, 1:5] = pred_boxes[i][:, 0:4]

        logit = logits[i]
        embed = outputs["last_hidden_state"][i]
        input = torch.concat([pos, logit, embed], dim=1)
        ret.append(input)

        targets = torch.zeros((num_obj), dtype=torch.long).to(device)
        targets.fill_(empty)
        targets[match_res[i][0]] = label["class_labels"][match_res[i][1]]
        box_mask = torch.zeros((num_obj), dtype=torch.long).to(device)
        box_mask.fill_(-1)
        if len(match_res[i][0]) > 0:
            t = torch.arange(0, len(match_res[i][0])).long().to(device)
            box_mask[match_res[i][0]] = t + cnts
            cnts += len(match_res[i][0])
        if need_mask and "masks" in label:
            mask.append(label["masks"][match_res[i][1]])
        boxes = torch.zeros((num_obj, 4)).to(device)
        boxes[match_res[i][0]] = label["boxes"][match_res[i][1]]

        item_id = ["" for i in range(num_obj)]
        if "item_id" in label:
            for s, t in zip(match_res[i][0], match_res[i][1]):
                item_id[s] = label["item_id"][t]

        boxeses.append(boxes)
        l.append(targets)
        item_ids.append(item_id)
        box_masks.append(box_mask)
    ret = torch.cat(ret, dim=0)
    l = torch.cat(l, dim=0)
    box_masks = torch.cat(box_masks, dim=0)
    boxeses = torch.cat(boxeses, dim=0)
    item_ids = [i for j in item_ids for i in j]
    if need_mask:
        mask = torch.cat(mask, dim=0)
        # print(cnts, mask.shape)
        return {
            "feature": ret,
            "label": l,
            "box_mask": box_masks,
            "boxes": boxeses,
            "item_id": item_ids,
            "masks": mask,
        }
    return {
        "feature": ret,
        "label": l,
        "box_mask": box_masks,
        "boxes": boxeses,
        "item_id": item_ids,
    }


from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from transformers.image_transforms import center_to_corners_format

import modules


def get_iou(X):
    iou, _ = modules.box_iou(
        center_to_corners_format(torch.tensor(X)),
        center_to_corners_format(torch.tensor(X)),
    )
    iou = iou.numpy()
    return iou


def get_neighbors(X, z_thres1=0.03, z_thres2=0.5, iou_thres=0.4):
    X2 = X[:, 1:5].clone()
    X2[:, 3:] += 0.05
    iou, _ = modules.box_iou(center_to_corners_format(X2), center_to_corners_format(X2))
    x, y = torch.where(iou > 0.01)
    zposx = X[:, 0][x]
    zposy = X[:, 0][y]
    x = x[torch.abs(zposx - zposy) < z_thres1]
    y = y[torch.abs(zposx - zposy) < z_thres1]

    x1, y1 = torch.where(iou > iou_thres)
    zposx = X[:, 0][x1]
    zposy = X[:, 0][y1]
    need = (zposx != zposy) & (torch.abs(zposx - zposy) < z_thres2)
    x1 = x1[need]
    y1 = y1[need]
    x = torch.cat([x, x1])
    y = torch.cat([y, y1])
    return x, y

    # dis = 1 - iou
    # t = t.numpy().copy()
    # t[:, 0] *= z
    # t[:, 1] *= x
    # # print(t, y)
    # t[:, 2] *= y
    # t[:, 3] *= x
    # t[:, 4] *= y
    # xs = []
    # ys = []
    # for i in range(t.shape[0] - 1):
    #     for j in range(i + 1, t.shape[0]):
    #         # d = np.linalg.norm(t[i, :3] - t[j, :3])
    #         # d2 = np.linalg.norm(t[i, 3:5])
    #         # d3 = np.linalg.norm(t[j, 3:5])
    #         # d1 = max(d2, d3) / 2
    #         if np.abs(t[i, 0] - t[j, 0]) < 0.00001:
    #             if dis[i, j] < 0.99:
    #                 xs.append(i)
    #                 ys.append(j)
    #                 xs.append(j)
    #                 ys.append(i)
    #         else:
    #             if dis[i, j] + np.abs(t[i, 0] - t[j, 0]) < radius:
    #                 xs.append(i)
    #                 ys.append(j)
    #                 xs.append(j)
    #                 ys.append(i)
    # # radius_neighbors = NearestNeighbors(radius=radius)
    # # radius_neighbors.fit(t)
    # # neighbors = radius_neighbors.radius_neighbors(t, return_distance=False)
    # return xs, ys


def convertStage2Dataset(retdict, z_thres1=0.01, z_thres2=0.2, iou_thres=0.4):
    X = retdict["feature"]

    # print("building up neighboring graph")
    xs, ys = get_neighbors(
        X.clone().detach(), z_thres1=z_thres1, z_thres2=z_thres2, iou_thres=iou_thres
    )
    # print(xs, ys, len(xs), len(ys))

    # print("building up neighboring graph done")

    y = retdict["label"]
    box_masks = retdict["box_mask"]
    boxes = retdict["boxes"]
    item_ids = retdict["item_id"]

    item_mapping = {"": -1}
    for i in item_ids:
        if i not in item_mapping:
            item_mapping[i] = len(item_mapping)
    item_ids = [item_mapping[i] for i in item_ids]
    item_ids = torch.tensor(item_ids, dtype=torch.long).to(X.device)
    item_idx = item_ids[xs]
    item_idy = item_ids[ys]
    edge_label = (item_idx == item_idy) & (item_idx != -1)
    edge_label = edge_label.long()
    # source = []
    # dest = []
    # edge_label = []
    # for i, j in zip(xs, ys):
    #     source.append(i)
    #     dest.append(j)
    #     if item_ids[i] == item_ids[j] and item_ids[i] != "":
    #         edge_label.append(1)
    #     else:
    #         edge_label.append(0)
    edges = torch.stack([xs, ys])
    # print(edges.shape)

    t = Data(x=X, edge_index=edges, y=y)
    t.box_masks = box_masks
    t.boxes = boxes
    t.edge_label = edge_label  # torch.tensor(edge_label, dtype=torch.long)
    if "sample_mapping" in retdict:
        t.sample_mapping = torch.tensor(
            [i for i in retdict["sample_mapping"].values()], dtype=int
        )
    return t


def get_stage2_model_embeddings(config_path, dataset, checkpoint_path="last.ckpt"):
    model = loadModel(config_path, checkpoint_path)

    ds = torch.load(dataset)

    with torch.no_grad():
        model.eval()
        res = model(x=ds.x, edge_index=ds.edge_index)
        # for i in ds:
        #     i = i.to("cuda")
        #     model(i.x, i.edge_index, i.train_mask)
        #     model(i.x, i.edge_index, i.val_mask)
    return res


def get_stage12_dataset(configs):
    if configs["data"]["transform"] == "default":
        t = getDefaultTransform()
    elif configs["data"]["transform"] == "simple":
        t = getSimpleTransform()
    else:
        t = getConstantTransform()

    if "norm" not in configs["data"]:
        configs["data"]["norm"] = "none"

    if "num" not in configs["data"]:
        configs["data"]["num"] = 5

    # dataset = modules.CocoDetection(
    #     configs["image_path"],
    #     configs["annotation_path"],
    #     is_npy=configs["is_npy"],
    #     transform=t,
    #     require_mask=configs["is_segmentation"],
    # )  # , transform=transforms)
    # train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    if isinstance(configs["data"]["annotation_path_train"], list):
        train_sets = []
        for i in configs["data"]["annotation_path_train"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            train_sets.append(
                modules.CocoDetection2(
                    configs["data"]["num"],
                    configs["data"]["image_path"],
                    i,
                    is_npy=configs["data"]["is_npy"],
                    transform=t,
                    require_mask=configs["data"]["require_mask"],
                    filter_class=configs["data"]["filter_class"],
                    single_class=configs["data"]["single_class"],
                    norm=configs["data"]["norm"],
                    map_class=map_class,
                )
            )
        train_sets = torch.utils.data.ConcatDataset(train_sets)
    else:
        map_class = None
        if "map_class" in configs["data"]:
            map_class = configs["data"]["map_class"]
        train_sets = modules.CocoDetection2(
            configs["data"]["num"],
            configs["data"]["image_path"],
            configs["data"]["annotation_path_train"],
            is_npy=configs["data"]["is_npy"],
            transform=t,
            require_mask=configs["data"]["require_mask"],
            filter_class=configs["data"]["filter_class"],
            single_class=configs["data"]["single_class"],
            norm=configs["data"]["norm"],
            map_class=map_class,
        )

    if isinstance(configs["data"]["annotation_path_val"], list):
        val_sets = []
        for i in configs["data"]["annotation_path_val"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            val_sets.append(
                modules.CocoDetection2(
                    configs["data"]["num"],
                    configs["data"]["image_path"],
                    i,
                    is_npy=configs["data"]["is_npy"],
                    transform=getConstantTransform(),
                    require_mask=configs["data"]["require_mask"],
                    filter_class=configs["data"]["filter_class"],
                    single_class=configs["data"]["single_class"],
                    norm=configs["data"]["norm"],
                    map_class=map_class,
                )
            )
        val_sets = torch.utils.data.ConcatDataset(val_sets)
    else:
        val_sets = modules.CocoDetection2(
            configs["data"]["num"],
            configs["data"]["image_path"],
            configs["data"]["annotation_path_val"],
            is_npy=configs["data"]["is_npy"],
            transform=getConstantTransform(),
            require_mask=configs["data"]["require_mask"],
            filter_class=configs["data"]["filter_class"],
            single_class=configs["data"]["single_class"],
            norm=configs["data"]["norm"],
            map_class=map_class,
        )

    ds = modules.EMDataModule(
        train_sets,
        val_sets,
        configs["training"]["train_batch_size"],
        configs["training"]["val_batch_size"],
        False,
    )
    # print("here build dataset stage 1")
    # print(ds)
    return ds


def get_stage1_dataset(configs):
    if configs["data"]["transform"] == "default":
        t = getDefaultTransform()
    elif configs["data"]["transform"] == "simple":
        t = getSimpleTransform()
    else:
        t = getConstantTransform()

    if "norm" not in configs["data"]:
        configs["data"]["norm"] = "none"

    if isinstance(configs["data"]["filter_class"], dict):
        train_sets = {}
        for i in configs["data"]["filter_class"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            train_sets[i] = modules.CocoDetection(
                configs["data"]["image_path"],
                configs["data"]["annotation_path_train"],
                is_npy=configs["data"]["is_npy"],
                transform=t,
                require_mask=configs["data"]["require_mask"][i],
                filter_class=configs["data"]["filter_class"][i],
                single_class=configs["data"]["single_class"][i],
                norm=configs["data"]["norm"],
                map_class=map_class,
                mark=i,
            )

        val_sets = {}
        for i in configs["data"]["filter_class"]:
            map_class = None
            if "map_class" in configs["data"]:
                map_class = configs["data"]["map_class"]
            val_sets[i] = modules.CocoDetection(
                configs["data"]["image_path"],
                configs["data"]["annotation_path_val"],
                is_npy=configs["data"]["is_npy"],
                transform=getConstantTransform(),
                require_mask=configs["data"]["require_mask"][i],
                filter_class=configs["data"]["filter_class"][i],
                single_class=configs["data"]["single_class"][i],
                norm=configs["data"]["norm"],
                map_class=map_class,
                mark=i,
            )
    else:
        map_class = None
        if "map_class" in configs["data"]:
            map_class = configs["data"]["map_class"]
        train_sets = modules.CocoDetection(
            configs["data"]["image_path"],
            configs["data"]["annotation_path_train"],
            is_npy=configs["data"]["is_npy"],
            transform=t,
            require_mask=configs["data"]["require_mask"],
            filter_class=configs["data"]["filter_class"],
            single_class=configs["data"]["single_class"],
            norm=configs["data"]["norm"],
            map_class=map_class,
        )
        val_sets = modules.CocoDetection(
            configs["data"]["image_path"],
            configs["data"]["annotation_path_val"],
            is_npy=configs["data"]["is_npy"],
            transform=getConstantTransform(),
            require_mask=configs["data"]["require_mask"],
            filter_class=configs["data"]["filter_class"],
            single_class=configs["data"]["single_class"],
            norm=configs["data"]["norm"],
            map_class=map_class,
        )

    # train_set, _val_set = torch.utils.data.random_split(dataset1, [0.8, 0.2])
    # _train_set, val_set = torch.utils.data.random_split(dataset2, [0.8, 0.2])
    # val_set.indices = _val_set.indices
    # trainloader = DataLoader(
    #     dataset=train_set,
    #     collate_fn=utils.stackBatch,
    #     batch_size=configs["training"]["train_batch_size"],
    #     shuffle=True,
    # )
    # valloader = DataLoader(
    #     dataset=val_set,
    #     collate_fn=utils.stackBatch,
    #     batch_size=configs["training"]["val_batch_size"],
    # )
    # testloader = DataLoader(dataset=val_set, collate_fn=utils.stackBatch, batch_size=1)

    ds = modules.EMDataModule(
        train_sets,
        val_sets,
        configs["training"]["train_batch_size"],
        configs["training"]["val_batch_size"],
    )
    # print("here build dataset stage 1")
    # print(ds)
    return ds


def get_stage2_dataset(configs):
    data_list_train = []
    for i in configs["data"]["train"]:
        data_list_train.append(torch.load(i))

    data_list_val = []
    for i in configs["data"]["val"]:
        data_list_val.append(torch.load(i))

    if "aug" not in configs["data"]:
        configs["data"]["aug"] = True

    ds = modules.stage2DataModule(
        data_list_train,
        data_list_val,
        configs["data"]["dataset_len"],
        ifaug=configs["data"]["aug"],
    )

    return ds


def get_stageMask_dataset(configs):
    pixels = []
    embeds = []
    masks = []
    sample_mapping = {}
    num1 = 0
    num2 = 0
    for i in configs["data"]["datasets"]:
        data = torch.load(i)
        pixels.extend(data["pixel_values"])
        embeds.append(data["embed"])
        masks.append(data["masks"])
        for j in data["sample_mapping"]:
            sample_mapping[j + num1] = data["sample_mapping"][j] + num2
        num1 += len(data["sample_mapping"])
        num2 += len(data["pixel_values"])
        # sample_mapping.update(data["sample_mapping"])
    pixels = torch.stack(pixels, dim=0)
    embeds = torch.cat(embeds, dim=0)
    masks = torch.cat(masks, dim=0)
    ds = modules.stage2MaskDataModule(
        pixels,
        embeds,
        masks,
        sample_mapping,
        batch_size=configs["training"]["train_batch_size"],
    )
    return ds


def loadModel(path, checkpoint="last.ckpt"):
    with open(os.path.join(path, "config.json"), "r") as f:
        configs = json.load(f)

    model = getModel(configs)

    ckpt = torch.load(os.path.join(path, checkpoint), map_location="cpu")["state_dict"]

    p = list(model.state_dict().keys())  # [j for j, _ in model.named_parameters()]
    # print("model parameters ", p)
    need_del = []
    for i in ckpt:
        if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
            need_del.append(i)
    for i in need_del:
        del ckpt[i]
    print("incompatible parameters", need_del)
    model.load_state_dict(ckpt, strict=False)
    print("finish loading parameters")

    # need_del = []
    # p = [j for j, _ in model.named_parameters()]
    # for i in ckpt:
    #     if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
    #         need_del.append(i)
    # for i in need_del:
    #     del ckpt[i]
    # model.load_state_dict(ckpt, strict=False)

    return model


def prepareMaskDataset(retdict, resdict):
    masks = retdict["masks"]
    images = retdict["images"]
    images = torch.stack(images, dim=0)
    embeds = resdict["embeddings"][retdict["box_mask"]]
    sapple_mapping = retdict["sample_mapping"]
    return {
        "embed": embeds,
        "pixel_values": images,
        "masks": masks,
        "sample_mapping": sapple_mapping,
    }


def generateMasks(model, embed, image, pixel_mask=None, batch_size=5, device="cpu"):
    model.stage = "stage mask"
    model = model.to(device)
    assert image.dim() == 3
    assert embed.dim() == 2
    res = []
    image = image[None, :, :, :].repeat(batch_size, 1, 1, 1)
    if pixel_mask is not None:
        pixel_mask = pixel_mask[None, :, :, :].repeat(batch_size, 1, 1, 1)
        pixel_mask = pixel_mask.to(device)
    image = image.float().to(device)
    embed = embed.to(device)
    for i in tqdm(range(0, embed.shape[0], batch_size)):
        with torch.no_grad():
            input_embed = embed[i : i + batch_size]
            res.append(
                model(
                    pixel_values=image[: input_embed.shape[0]].float(),
                    pixel_mask=pixel_mask,
                    stage_2_embeds=input_embed,
                )
            )
    res = torch.cat(res, dim=0)
    res = res.cpu()
    model = model.cpu()
    return res


import numpy as np


def calculate_iou(pred_bbox, gt_bbox):
    """Calculate Intersection over Union (IoU) between predicted and ground truth bounding boxes."""
    # Compute the coordinates of the intersection rectangle
    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])

    # Compute the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute the area of both bounding boxes
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    # Compute the area of the union
    union_area = pred_area + gt_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou


def calculate_precision_recall(pred_bboxes_list, gt_bboxes_list, iou_threshold=0.5):
    """Calculate Precision and Recall for a given IoU threshold."""
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    matched_gt = []  # Keep track of which ground truth boxes are matched
    for pred_bboxes, gt_bboxes in zip(pred_bboxes_list, gt_bboxes_list):
        for pred_bbox in pred_bboxes:
            ious = [calculate_iou(pred_bbox, gt_bbox) for gt_bbox in gt_bboxes]
            max_iou = max(ious)

            if max_iou >= iou_threshold:
                idx = ious.index(max_iou)
                if idx not in matched_gt:
                    tp += 1
                    matched_gt.append(idx)
                # tp += 1
                # matched_gt.append(ious.index(max_iou))
            else:
                fp += 1

    fn = len(gt_bboxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def calculate_ap(pred_bboxes, gt_bboxes, iou_thresholds=[0.5, 0.75]):
    """Calculate AP at different IoU thresholds (AP50, AP75)."""
    aps = {}
    for iou_threshold in iou_thresholds:
        precision, recall = calculate_precision_recall(
            pred_bboxes, gt_bboxes, iou_threshold
        )
        ap = (
            precision * recall
        )  # Simplified AP calculation (you may use interpolation methods)
        aps[f"AP@{iou_threshold*100}"] = ap

    return aps
