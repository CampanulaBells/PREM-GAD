import numpy as np
import torch
import time

from sklearn.metrics._ranking import roc_auc_score
from tqdm import tqdm
from modules.model import Model
from modules.train import train_model, eval_model
from modules.utils import load_dataset, set_random_seeds, rescale

def run_experiment(args, seed, device, dataloader, ano_label):
    set_random_seeds(seed)
    # Create GGD model
    model = Model(
        g=dataloader.g,
        n_in=dataloader.en.shape[1],
        n_hidden=args.n_hidden,
        k=args.k
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_function = torch.nn.BCELoss()

    print(f"Seed {seed}")
    torch.cuda.reset_peak_memory_stats()
    state_path, stats, time_train = train_model(
        args, dataloader, model, optimizer, loss_function
    )
    mem_train = torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(state_path))
    torch.cuda.reset_peak_memory_stats()
    score, time_test = eval_model(args, dataloader, model, ano_label)
    mem_test = torch.cuda.max_memory_allocated()

    auc = roc_auc_score(ano_label, score)
    stats["mem_train"] = mem_train
    stats["mem_test"] = mem_test
    stats["time_train"] = time_train
    stats["time_test"] = time_test
    stats["AUC"] = auc

    return model, stats

