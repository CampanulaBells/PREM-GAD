
# ICDM23 PREM: A Simple Yet Effective Approach for Node-Level Graph Anomaly Detection

Junjun Pan, Yixin Liu, Yizhen Zheng, Shirui Pan
---
This repo contains the official implementation of [ICDM23 PREM: A Simple Yet Effective Approach for Node-Level Graph Anomaly Detection](https://arxiv.org/abs/2310.11676)

<img src="./assets/Architecture.png"
     style="float: left; margin-right: 10px;" />
     
To reproduce the results proposed in the paper, run 

### Cora

```
python run.py --dataset cora --lr 0.0003 --alpha 0.9 --gamma 0.1 --num_epoch 100
```

### Citeseer

```
python run.py --dataset citeseer --lr 0.0003 --alpha 0.9 --gamma 0.1 --num_epoch 100
```

### PubMed

```
python run.py --dataset pubmed --lr 0.0005 --alpha 0.6 --gamma 0.4 --num_epoch 400
```

### ACM

```
python run.py --dataset ACM --lr 0.0001 --alpha 0.7 --gamma 0.2 --num_epoch 200
```

### Flickr

```
python run.py --dataset Flickr --lr 0.0005 --alpha 0.3 --gamma 0.4 --num_epoch 1500
```

---


### Environment

The code is tested under conda environment (py 3.7.15) with these additional libs installed: 

Please let us know if you find other libs are also required. 

```
dgl==1.0.0+cu113
torch==1.12.0+cu113
torch-geometric==2.3.1
torch-scatter==2.1.1
torch-sparse==0.6.17
torch-spline-conv==1.2.2
tqdm==4.64.1
```



---


If you find our work useful in your research, please consider citing:

```
@inproceedings{pan2023prem,
  title={PREM: A Simple Yet Effective Approach for Node-Level Graph Anomaly Detection},
  author={Pan, Junjun and Liu, Yixin and Zheng, Yizhen and Pan, Shirui},
  booktitle={2023 IEEE International Conference on Data Mining (ICDM)},
  pages={1253--1258},
  year={2023},
  organization={IEEE}
}
```

