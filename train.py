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
from transformers import DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor

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
    CHECKPOINT = "facebook/detr-resnet-50"
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

    dataset = utils.CocoDetection(
        configs["image_path"],
        configs["annotation_path"],
        image_processor,
        is_npy=configs["is_npy"],
    )  # , transform=transforms)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    trainloader = DataLoader(
        dataset=train_set,
        collate_fn=utils.get_collate_fn(image_processor),
        batch_size=configs["batch_size"],
        shuffle=True,
    )
    valloader = DataLoader(
        dataset=val_set,
        collate_fn=utils.get_collate_fn(image_processor),
        batch_size=configs["batch_size"],
    )
    testloader = DataLoader(
        dataset=val_set, collate_fn=utils.get_collate_fn(image_processor), batch_size=1
    )

    ds = utils.EMDataModule(trainloader, valloader, testloader)

    print("finish loading data")

    print("building model")

    categories = dataset.coco.cats
    id2label = {k: v["name"] for k, v in categories.items()}

    if configs["is_segmentation"]:

        pretrain = DetrForSegmentation.from_pretrained(
            pretrained_model_name_or_path="facebook/detr-resnet-50",
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )
    else:
        pretrain = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path="facebook/detr-resnet-50",
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )

    model = utils.Detr(
        lr=configs["lr"],
        lr_backbone=configs["lr_backbone"],
        weight_decay=configs["weight_decay"],
        model=pretrain,
    )

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
        max_epochs=configs["epoch"],
        gradient_clip_val=0.1,
        accumulate_grad_batches=configs["accumulate_grad_batches"],
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )

    print("start training")
    trainer.fit(model, ds)


if __name__ == "__main__":
    run()
