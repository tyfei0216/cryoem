import argparse
import json
import os

import numpy as np
import pytorch_lightning as L
import tensorboard
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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

    data_list = []
    for i in configs["data"]["graphs"]:
        data_list.append(torch.load(i))

    ds = modules.stage2DataModule(
        data_list,
        configs["data"]["dataset_len"],
    )

    print("finish loading data")

    print("building model")

    model = utils.getModel(configs)

    # if "checkpoint" in configs["model"] and configs["model"]["checkpoint"] is not None:
    #     t = torch.load(configs["model"]["checkpoint"], map_location="cpu")
    #     model.load_state_dict(t["state_dict"], strict=False)

    print("finish build model")

    checkpoint_callback = ModelCheckpoint(
        monitor="validate_acc",  # Replace with your validation metric
        mode="max",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
        save_top_k=3,  # Save top k checkpoints based on the monitored metric
        save_last=True,  # Save the last checkpoint at the end of training
        dirpath=args.path,  # Directory where the checkpoints will be saved
        filename="{epoch}-{validate_loss:.2f}-{validate_acc:.4f}",  # Checkpoint file naming pattern
    )
    if "gradient_clip_val" not in configs["training"]:
        configs["training"]["gradient_clip_val"] = None

    logger_path = (
        configs["training"]["logger_path"]
        if "logger_path" in configs["training"]
        else "tb_logs"
    )
    name = (
        configs["training"]["name"]
        if "name" in configs["training"]
        else os.path.basename(args.path)
    )

    logger = TensorBoardLogger(logger_path, name=name)
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
