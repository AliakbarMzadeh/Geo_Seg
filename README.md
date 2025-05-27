# Segmentation and Vascular Vectorization for Coronary Artery by Geometry-based Cascaded Neural Network



## Abstract

Segmentation of the coronary artery, angiography (CCTA). geometry-based segmentation network. 
## Network

![workflow of our geometry-based Cascaded Neural Network](./images/workflow.jpg)

> Fig.1  geometry-based cascaded segmentation network for generating mesh of the coronary artery.


![workflow of our geometry-based Cascaded Neural Network](./images/Tabel.png)
> Fig.1  Fine Mesh Annotation for Geometrical Regularization

## Graph Convolutional Network
Graph Convolutional Network: a sphere mesh $\mathcal{G}=\{\mathcal{V}, \mathcal{E}\}$ with 162 vertices and 480 edges is initialized as the input of the GCN, where $\mathcal{V}$ denotes the set of vertices and $\mathcal{E}$ represents the set of edges. The mesh with $N$ vertices $v_i \in \mathcal{V}$ in the GCN has its adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ and diagonal degree
$\hat{\mathbf{D}}_{i i}=\sum_{j=0} \hat{\mathbf{A}}_{i j}$, where $\hat{\mathbf{A}}=\mathbf{A}+\mathbf{I}$. The graph convolution is executed as Eq. 1.

$$
\mathbf{V}^{\prime}=\hat{\mathbf{D}}^{-1 / 2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1 / 2} \mathbf{V} \Theta
$$

where $\Theta$ represents the parameters of the neural network and $\mathbf{V} \in \mathbb{R}^{N \times C}$ symbolizes the feature vector with $C$ dimension for each node $v_i$. In addition, the residual block is applied to predict the deformation of the mesh instead of predicting the vertices location of the target mesh directly, which simplifies the difficulty of training. Furthermore, the initial sphere is easily deformed but lacks enough details of the coronary artery. Graph unpooling is implemented in our GCN at stage I, dividing one triangular face into four parts along the midpoint of each side and assigning the mean feature vector of one edge to the node of the midpoint. It supplements more vertices and edges, retouching the mesh of the coronary artery. The LNS [24] strategy is performed to project extracted image features into the mesh space.






## Optimization of Segmentation Network

Optimization of Segmentation Network: For jointly training the U-shape neural networks and GCN, various loss functions are adopted to optimize them. First, image loss is mainly driving the U-shape network under the voxel-based segmentation framework, consisting of SoftDice loss and cross-entropy loss. Second, mesh loss optimizes the GCN, including chamfer distance loss, laplacian smoothing, normal consistency loss and edge loss. The chamfer distance dominates the optimization of the GCN, which measures the distance of two point clouds between the prediction and ground truth as Eq.2, guiding the deformation of the mesh.

```
L_CD(V1, V2) = (1 / |V1|) ∑_{x ∈ V1} min_{y ∈ V2} ||x - y||²
             + (1 / |V2|) ∑_{y ∈ V2} min_{x ∈ V1} ||x - y||²
```

Laplacian smoothing (Lap) and normal consistency loss (NC) are utilized to regularize the smoothness of the mesh. Laplacian smoothing L_Lap computes the uniform weights of all edges connected at a vertex. Normal consistency loss computes the angle of the normal n₀ and n₁ for each pair of neighboring faces as Eq. 3.

```
L_NC = ∑_{e ∈ E} (1 - cos(n₀, n₁))
```

Besides, edge loss L_EG computes the length of each edge, avoiding outlier vertices. In summary, the total loss of the GCN is shown in Eq. 4.

```
L_GCN = λ₁ * L_CD + λ₂ * L_Lap + λ₃ * L_NC + λ₄ * L_EG
```




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
