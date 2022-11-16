# CorePPR 

This repository provides the reference implementation of CorePPR for a single machine in TensorFlow 1. 

CorePPR is a scalable model that utilises a learnable convex combination of the approximate personalised PageRank and the CoreRank to diffuse multi-hop
neighbourhood information in GNNs. It was proposed in our paper

**[Improving Graph Neural Networks at Scale: Combining Approximate PageRank and CoreRank](https://arxiv.org/abs/2211.04248)**   
by Ariel R. Ramos Vela\*, Johannes F. Lutzeyer\*, Anastasios Giovanidis, Michalis Vazirgiannis 
Accepted at the "NeurIPS 2022 New Frontiers in Graph Learning Workshop (NeurIPS GLFrontiers 2022)"

\*Corresponding authors:


## Installation
You can install the repository using `pip install -e .`. However, installing the requirements like this will result in TensorFlow using CUDA 10.0, which contains a bug that affects PPRGo. We recommend importing the Anaconda environment saved in `environment.yaml` instead, which provides the correct TensorFlow and CUDA versions.

## Run the code
To see for yourself how CorePPR performs on a large dataset, we have included a (`demo.ipynb`) notebook that trains and generates predictions for the datasets described in the paper.

## Contact
Please contact ariel.ramosvela@ip-paris.fr or johannes.lutzeyer@polytechnique.edu. if you have any question.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@article{vela2022improving,
  title={Improving Graph Neural Networks at Scale: Combining Approximate PageRank and CoreRank},
  author={Vela, Ariel R Ramos and Lutzeyer, Johannes F and Giovanidis, Anastasios and Vazirgiannis, Michalis},
  journal={arXiv preprint arXiv:2211.04248},
  year={2022}
}
```
