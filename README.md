# Robotic nanofabrication

## Introduction
This repository contains the code for the Reinforcement Learning agent described in the paper

### Autonomous robotic nanofabrication with reinforcement learning

Science Advances (open access): https://advances.sciencemag.org/content/6/36/eabb6987.full

Arxiv: https://arxiv.org/abs/2002.11952

## Usage

In order to install the requirements (using conda):

```bash
conda env create -f conda_environment.yaml
conda activate robotic_nanofab
```

In order to run the RL algorithm in the simulated environment, do

```bash
python robotic_nanofabrication/main.py --output_path=outfiles/test1
```

## Recreate swarm plots in paper
The swarm plots in the paper (Fig. 2D) have been created using the follwing repository:

https://github.com/maltimore/gridsearch_helper

you should use tag `0.0.2`.

That repository has the scripts to parallelize experiments on the Sun Grid Engine or Univa Grid Engine. Additionally, it contains python files that can aggregate the data from the parallelized runs and create the swarm plots.
