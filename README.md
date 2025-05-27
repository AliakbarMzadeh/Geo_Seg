# Segmentation and Vascular Vectorization for Coronary Artery by Geometry-based Cascaded Neural Network



## Abstract

Segmentation of the coronary artery, angiography (CCTA). geometry-based segmentation network. 
## Network

![workflow of our geometry-based Cascaded Neural Network](./images/workflow.jpg)

> Fig.1  geometry-based cascaded segmentation network for generating mesh of the coronary artery.





## Installation

PyTorch == 1.11.0

Python == 3.9.12

torch-geometric == 2.1.0

pytorch3d == 0.7.0

pyvista == 0.36.1

trimesh == 3.12.6

## Experiments

train on stage I:

```bash
nohup python -u ./train.py -c "./config/config-s1-train.yaml" > train-s1.log 2>&1 &
```

train on stage II:

```bash
nohup python -u ./train.py -c "./config/config-s2-train.yaml" > train-s2.log 2>&1 &
```

predict:

```bash
nohup python -u ./predict.py -c "./config/config-predict.yaml" > predict.log 2>&1 &
```
