import importlib
import json
import os

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

import modules
import Unet
import utils

checkpoint_callback2 = ModelCheckpoint(
    every_n_epochs=10,
    save_top_k=-1,
    save_last=False,  # Save the last checkpoint at the end of training
    dirpath="./temp",  # Directory where the checkpoints will be saved
    filename="{epoch}",  # Checkpoint file naming pattern
)

trainer = L.Trainer(devices=[0, 1], max_epochs=80, callbacks=[checkpoint_callback2])
model = Unet.tryUnet()

path = "/data/transformer_project/transforemer_model/train_data/training/results/conditional_detr/train_mask"
with open(os.path.join(path, "config.json"), "r") as f:
    configs = json.load(f)
ds = utils.get_stageMask_dataset(configs)

trainer.fit(model, ds)
