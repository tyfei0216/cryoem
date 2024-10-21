import sys

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrForSegmentation,
    DetrConfig,
    DetrForObjectDetection,
    DetrForSegmentation,
    DetrImageProcessor,
)

sys.path.append("/home/feity/cyroem")
import modules
import utils

torch.manual_seed(1509)
dataset1 = modules.CocoDetection(
    "/home/feity/cryoem/dataset/Zscore",
    "/home/feity/cryoem/dataset/npy_annotations.pkl",
    is_npy=True,
    transform=utils.getDefaultTransform(),
    require_mask=True,
    filter_class=[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    # single_class=True,
)
dataset2 = modules.CocoDetection(
    "/home/feity/cryoem/dataset/Zscore",
    "/home/feity/cryoem/dataset/npy_annotations.pkl",
    is_npy=True,
    transform=utils.getConstantTransform(),
    require_mask=True,
    filter_class=[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    # single_class=True,
)
train_set, _val_set = torch.utils.data.random_split(dataset1, [0.8, 0.2])
_train_set, val_set = torch.utils.data.random_split(dataset1, [0.8, 0.2])
val_set.indices = _val_set.indices
categories = dataset1.coco.cats
id2label = {k: v["name"] for k, v in categories.items()}

trainloader = DataLoader(
    dataset=train_set,
    collate_fn=utils.stackBatch,
    batch_size=2,
    shuffle=True,
    num_workers=4,
)
valloader = DataLoader(
    dataset=val_set, collate_fn=utils.stackBatch, batch_size=2, num_workers=4
)
testloader = DataLoader(dataset=val_set, collate_fn=utils.stackBatch, batch_size=1)
ds = modules.EMDataModule(trainloader, valloader, testloader)

configs = ConditionalDetrConfig(num_labels=len(id2label))
seg_model = ConditionalDetrForSegmentation(configs)
model = ConditionalDetrForObjectDetection.from_pretrained(
    "microsoft/conditional-detr-resnet-50",
    num_labels=len(id2label),
    ignore_mismatched_sizes=True,
)
seg_model.conditional_detr.load_state_dict(model.state_dict())

model = modules.Detr(lr=5e-5, model=model, lr_backbone=1e-5, weight_decay=1e-5)
# model = model.to(7)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")
pl.seed_everything(1509)
# %cd {HOME}
# settings
MAX_EPOCHS = 7200

checkpoint_callback = ModelCheckpoint(
    monitor="validate_loss",  # Replace with your validation metric
    mode="min",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
    save_top_k=3,  # Save top k checkpoints based on the monitored metric
    save_last=True,  # Save the last checkpoint at the end of training
    dirpath="/home/feity/cryoem/checkpoints/conditional_detr_without3_transform_7200_zscore",  # Directory where the checkpoints will be saved
    filename="{epoch}-{validate_loss:.2f}",  # Checkpoint file naming pattern
)
# pytorch_lightning >= 2.0.0
logger = TensorBoardLogger("tb_logs", "detr_seg_3")
trainer = Trainer(
    devices=[0],
    accelerator="gpu",
    max_epochs=MAX_EPOCHS,
    gradient_clip_val=0.1,
    accumulate_grad_batches=4,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, ds)
