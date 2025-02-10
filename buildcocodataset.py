import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("/home/tyfei/cyroem")
import importlib

import utils

importlib.reload(utils)
import argparse

import pycocotools.coco
import pycocotools.mask


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
    pass


if __name__ == "__main__":
    run()
