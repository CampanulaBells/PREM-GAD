import time
from sklearn.metrics import roc_auc_score
import json
import os
import copy
import argparse
import numpy as np
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.model import Model, Dataloader
from modules.utils import load_dataset, set_random_seeds, rescale
from modules.experiment import run_experiment

parser = argparse.ArgumentParser(description='GGD Anomaly')
parser.add_argument('--dataset', type=str, default='Flickr')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--gamma', type=float, default=0.4)

parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--k', type=int, default=2)

parser.add_argument('--resultdir', type=str, default='results')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=1500)
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--batch_size', type=int, default=-1)

if __name__ == '__main__':
    args = parser.parse_args()
    # Setup torch
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Load dataset
    g, features, ano_label, _, _ = load_dataset(args.dataset)
    features = torch.FloatTensor(features)
    if args.batch_size == -1:
        features = features.to(device)
    g = g.to(device)
    dataloader = Dataloader(g, features, args.k, dataset_name=args.dataset)
    if not os.path.isdir("./ckpt"):
        os.makedirs("./ckpt")

    # Run the experiment
    seed = args.seed
    model, stats = run_experiment(args, seed, device, dataloader, ano_label)
    print("AUC: %.4f" % stats["AUC"])
    print("Time (Train): %.4fs" % stats["time_train"])
    print("Mem (Train): %.4f MB" % (stats["mem_train"] / 1024 / 1024))
    print("Time (Test): %.4fs" % stats["time_test"])
    print("Mem (Test): %.4f MB" % (stats["mem_test"] / 1024 / 1024))
    exit()
