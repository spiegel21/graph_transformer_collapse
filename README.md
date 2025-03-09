# Exploring neural collapse in graph neural networks

This repository contains the code for the DSC 180B Senior Capstone at UC San Diego. In this study, we seek to observe the presence of Neural Collapse in Graph Transformers. This repository is built upon the existing work by Vignesh Kothapalli, Tom Tirer, and Joan Bruna in their [paper](https://arxiv.org/abs/2307.01951) "A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks". The original repository for this work can be found [here](https://github.com/kvignesh1420/gnn_collapse/tree/main)

## Setup

torch needs to be installed before the rest of the modules in requirements.txt due to the nature of the setup for torch-scatter and torch-sparse. [See this thread]() for an explanation.

```bash
$ python3.12 -m virtualenv .venv
$ source .venv/bin/activate
$ pip install torch==2.5.1
$ pip install -r requirements.txt
```

## SBM Experiments

Experiments on SBMs are contained within the directory sbm_collapse. Begin by moving to this directory
```
cd sbm_collapse
```
Next, follow the steps written in the original repository:

We employ a config based design to run and hash the experiments. The `configs` folder contains the `final` folder to maintain the set of experiments that have been presented in the paper. The `experimental` folder is a placeholder for new contributions. A config file is a JSON formatted file which is passed to the python script for parsing. The config determines the runtime parameters of the experiment and is hashed for uniqueness.

To run GNN experiments:
```bash
$ bash run_gnn.sh
```

A new folder called `out` will be created and the results are stored in a folder named after the hash of the config.

## CORA Experiments

Experiments on CORA are all contained within the directory cora_collapse. Begin by moving to this directory
```
cd cora_collapse
```
From here, you may run our experiments with
```
python main.py --cfg configs/[cora-GCN.yaml or cora-GT-yaml] wandb.use False
```
This file will save neural collapse metrics locally. The code to generate the plots found in the paper is in `plotting_notebook.ipynb`

## MNIST Experiments

Experiments on MNIST (found in the appendix of our report) are contained within the directory mnist_collapse. Begin by moving to this directory
```
cd mnist_collapse
```
The MNIST code is split into three main files.
To train models and save parameters
```
python train.py [resnet18 or vit_small]
```
To collect last layer activations
```
python collect_activations.py [resnet18 or vit_small]
```
Lastly, plotting code can be found in the Jupyter Notebook `plot_collapse.ipynb`

## Acknowledgements

Code in the `sbm_collapse` directory is based on the GitHub repository [here](https://github.com/kvignesh1420/gnn_collapse), for the paper "A neural collapse perspective on feature evolution in graph neural networks" by Vignesh Kothapalli, Tom Tirer, and Joan Bruna.

Code in the `cora_collapse` directory is based on the Github repository [here](https://github.com/rampasek/GraphGPS), for the paper "Recipe for a General, Powerful, Scalable
Graph Transformer" by Ladislav Rampášek et al.

We'd also like to thank our faculty advisors, Yusu Wang and Gal Mishne, for their mentorship this quarter, Professor Umesh Bellur for organizing the data science capstone project, our TA Zimo Wang for his feedback throughout the quarter and our classmates Penny King, Esther Cho, and Quentin Callahan for their help in working with the GraphGPS code.
