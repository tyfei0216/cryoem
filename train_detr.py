import argparse
import json
import os

import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
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
import utils

torch.set_float32_matmul_precision("high")


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=[0])
    parser.add_argument("-s", "--strategy", type=str, default="auto")
    parser.add_argument("-n", "--name", type=str, default="detr")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args


def run():
    args = parseArgs()
    path = args.path

    with open(os.path.join(path, "config.json"), "r") as f:
        configs = json.load(f)
    json_formatted_str = json.dumps(configs, indent=2)
    print("fetch config from ", os.path.join(path, "config.json"))
    print("--------")
    print("config: ")
    print(json_formatted_str)
    print("--------")
    print("using devices ", args.devices)
    print("using strategy ", args.strategy)
    print("--------")

    L.seed_everything(configs["seed"])

    print("loading dataset")
    # CHECKPOINT = "facebook/detr-resnet-50"
    # image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

    if configs["data"]["transform"] == "default":
        t = utils.getDefaultTransform()
    else:
        t = utils.getConstantTransform()

    # dataset = modules.CocoDetection(
    #     configs["image_path"],
    #     configs["annotation_path"],
    #     is_npy=configs["is_npy"],
    #     transform=t,
    #     require_mask=configs["is_segmentation"],
    # )  # , transform=transforms)
    # train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    if "zscore" not in configs["data"]:
        configs["data"]["zscore"] = False

    train_set = modules.CocoDetection(
        configs["data"]["image_path"],
        configs["data"]["annotation_path_train"],
        is_npy=configs["data"]["is_npy"],
        transform=t,
        require_mask=configs["data"]["require_mask"],
        filter_class=configs["data"]["filter_class"],
        single_class=configs["data"]["single_class"],
        zscore=configs["data"]["zscore"],
    )
    val_set = modules.CocoDetection(
        configs["data"]["image_path"],
        configs["data"]["annotation_path_val"],
        is_npy=configs["data"]["is_npy"],
        transform=utils.getConstantTransform(),
        require_mask=configs["data"]["require_mask"],
        filter_class=configs["data"]["filter_class"],
        single_class=configs["data"]["single_class"],
        zscore=configs["data"]["zscore"],
    )
    # train_set, _val_set = torch.utils.data.random_split(dataset1, [0.8, 0.2])
    # _train_set, val_set = torch.utils.data.random_split(dataset2, [0.8, 0.2])
    # val_set.indices = _val_set.indices
    trainloader = DataLoader(
        dataset=train_set,
        collate_fn=utils.stackBatch,
        batch_size=configs["training"]["train_batch_size"],
        shuffle=True,
    )
    valloader = DataLoader(
        dataset=val_set,
        collate_fn=utils.stackBatch,
        batch_size=configs["training"]["val_batch_size"],
    )
    testloader = DataLoader(dataset=val_set, collate_fn=utils.stackBatch, batch_size=1)

    ds = modules.EMDataModule({"train": train_set}, {"val": val_set}, 2, 4)

    print("finish loading data")

    print("building model")

    # categories = dataset.coco.cats
    # id2label = {k: v["name"] for k, v in categories.items()}

    # if not configs["use_pretrained"]:
    #     configs["lr_backbone"] = None

    if configs["model"]["name"] == "conditional_detr":
        if configs["model"]["task"] == "segmentation":
            config = ConditionalDetrConfig(
                use_pretrained_backbone=False, **configs["model"]["args"]
            )
            seg_model = ConditionalDetrForSegmentation(config)
            if len(configs["model"]["pretrained"]) > 0:
                model = ConditionalDetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"],
                    ignore_mismatched_sizes=True,
                    **configs["model"]["args"]
                )
                seg_model.conditional_detr.load_state_dict(model.state_dict())
            model = seg_model
        elif configs["model"]["task"] == "detection":
            if len(configs["model"]["pretrained"]) > 0:
                model = ConditionalDetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"],
                    ignore_mismatched_sizes=True,
                    **configs["model"]["args"]
                )
            else:
                config = ConditionalDetrConfig(
                    use_pretrained_backbone=False, **configs["model"]["args"]
                )
                model = ConditionalDetrForObjectDetection(config)

    elif configs["model"]["name"] == "deformable_detr":
        if len(configs["model"]["pretrained"]) > 0:
            model = DeformableDetrForObjectDetection.from_pretrained(
                configs["model"]["pretrained"],
                ignore_mismatched_sizes=True,
                **configs["model"]["args"]
            )
        else:
            config = DeformableDetrConfig(
                use_pretrained_backbone=False, **configs["model"]["args"]
            )
            model = DeformableDetrForObjectDetection(config)

    elif configs["model"]["name"] == "detr":
        if configs["model"]["task"] == "segmentation":
            if len(configs["model"]["pretrained"]) > 0:
                model = DetrForSegmentation.from_pretrained(
                    configs["model"]["pretrained"],
                    ignore_mismatched_sizes=True,
                    **configs["model"]["args"]
                )
            else:
                config = DetrConfig(
                    use_pretrained_backbone=False, **configs["model"]["args"]
                )
                model = DetrForSegmentation(config)
        elif configs["model"]["task"] == "detection":
            if len(configs["model"]["pretrained"]) > 0:
                model = DetrForObjectDetection.from_pretrained(
                    configs["model"]["pretrained"],
                    ignore_mismatched_sizes=True,
                    **configs["model"]["args"]
                )
            else:
                config = DetrConfig(
                    use_pretrained_backbone=False, **configs["model"]["args"]
                )
                model = DetrForObjectDetection(config)
    else:
        raise NotImplementedError

    model = modules.Detr(
        lr=configs["training"]["lr"],
        lr_backbone=configs["training"]["lr_backbone"],
        weight_decay=configs["training"]["weight_decay"],
        model=model,
    )

    if "checkpoint" in configs["model"] and configs["model"]["checkpoint"] is not None:
        t = torch.load(configs["model"]["checkpoint"], map_location="cpu")
        model.load_state_dict(t["state_dict"], strict=False)

    print("finish build model")

    checkpoint_callback = ModelCheckpoint(
        monitor="validate_loss",  # Replace with your validation metric
        mode="min",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
        save_top_k=3,  # Save top k checkpoints based on the monitored metric
        save_last=True,  # Save the last checkpoint at the end of training
        dirpath=args.path,  # Directory where the checkpoints will be saved
        filename="{epoch}-{validate_loss:.2f}",  # Checkpoint file naming pattern
    )

    logger = TensorBoardLogger("tb_logs", args.name)
    trainer = Trainer(
        logger=logger,
        devices=args.devices,
        accelerator="gpu",
        max_epochs=configs["training"]["epochs"],
        gradient_clip_val=configs["training"]["gradient_clip_val"],
        accumulate_grad_batches=configs["training"]["accumulate_grad_batches"],
        log_every_n_steps=configs["training"]["log_every_n_steps"],
        callbacks=[checkpoint_callback],
    )

    print("start training")
    if args.checkpoint is not None:
        trainer.fit(model, ds, ckpt_path=args.checkpoint)
    else:
        trainer.fit(model, ds)
    print("finish training")


if __name__ == "__main__":
    run()
