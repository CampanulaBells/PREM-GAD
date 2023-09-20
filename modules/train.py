import time
from sklearn.metrics import roc_auc_score
import json
import copy
import os
import numpy as np
import torch
from modules.utils import set_random_seeds, rescale

def train_model(args, dataloader, model, optimizer, loss_function):
    stats = {
        "best_loss": 1e9,
        "best_epoch": -1,
    }
    state_path = f'./ckpt/{args.dataset}.pkl'
    time_train = time.time()
    model.train()
    if args.batch_size > 0:
        label_ones =  dataloader.label_ones[:, 0:args.batch_size].to("cuda")
        label_zeros =  dataloader.label_zeros[:, 0:args.batch_size].to("cuda")
    for epoch in range(args.num_epoch):
        optimizer.zero_grad()
        en_p, en_n, eg_p, eg_aug = dataloader.get_data()
        # Full batch
        if args.batch_size == -1:
            # Full batch
            score_pos = rescale(model(en_p, eg_p))
            score_aug = rescale(model(en_p, eg_aug))
            score_nod = rescale(model(en_p, en_n))

            loss_pos = loss_function(score_pos, dataloader.label_zeros)
            loss_aug = loss_function(score_aug, dataloader.label_ones)
            loss_nod = loss_function(score_nod, dataloader.label_ones)

            loss_sum = loss_pos \
                   + args.alpha * loss_aug \
                   + args.gamma * loss_nod
            loss_sum.backward()
        else:
            i = 0
            loss_pos = 0
            while i * args.batch_size < len(en_p):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, len(en_p))
                en_p_batch, en_n_batch, eg_p_batch, eg_aug_batch = en_p[start_index:end_index], en_n[start_index:end_index], eg_p[start_index:end_index], eg_aug[start_index:end_index]
                en_p_batch, en_n_batch, eg_p_batch, eg_aug_batch = [x.to("cuda") for x in [ en_p_batch, en_n_batch, eg_p_batch, eg_aug_batch]]
                i += 1
                score_pos = rescale(model(en_p_batch, eg_p_batch))
                score_aug = rescale(model(en_p_batch, eg_aug_batch))
                score_nod = rescale(model(en_p_batch, en_n_batch))
                loss_pos_batch = loss_function(score_pos, label_zeros[:, :end_index-start_index])
                loss_aug_batch = loss_function(score_aug, label_ones[:, :end_index-start_index])
                loss_nod_batch = loss_function(score_nod, label_ones[:, :end_index-start_index])
                loss_sum_batch = loss_pos_batch \
                       + args.alpha * loss_aug_batch \
                       + args.gamma * loss_nod_batch
                # Rescale loss
                loss_sum_batch = loss_sum_batch * (end_index - start_index) / len(en_p)
                loss_sum_batch.backward()
                loss_pos += loss_pos_batch.item()

        if loss_pos < stats["best_loss"]:
            stats["best_loss"] = loss_pos
            stats["best_epoch"] = epoch
            torch.save(model.state_dict(), state_path)
        optimizer.step()

    time_train = time.time() - time_train
    return state_path, stats, time_train

def eval_model(args, dataloader, model, ano_label):
    model.eval()
    with torch.no_grad():
        time_test = time.time()
        if args.batch_size == -1:
            score = model(dataloader.en, dataloader.eg)[0].cpu().numpy()
        else:
            score = []
            en = dataloader.en
            eg = dataloader.eg
            i = 0
            while i * args.batch_size < len(en):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, len(en))
                en_batch, eg_batch = en[start_index:end_index], eg[start_index:end_index]
                en_batch, eg_batch = [x.to("cuda") for x in [en_batch, eg_batch]]
                score.append(model(en_batch, eg_batch).detach().cpu().numpy())
                i += 1
            score = np.concatenate(score, axis=1)[0]

        time_test = time.time() - time_test
    return score, time_test
