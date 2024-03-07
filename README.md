# Proteus
Proteus is an ML inference serving system for resource-constrained compute clusters that proposes a technique to handle workload spikes called "accuracy scaling". Accuracy scaling leverages the flexibility of ML models to adjust the accuracy of served inference requests to adapt to increased load, as opposed to traditional hardware scaling that adapts by adding hardware.

Proteus is set to appear in ASPLOS 2024. You can find it [here](https://guanh01.github.io/files/2024proteus.pdf).

This artifact describes the complete workflow to setup the simulation experiments for Proteus. We describe how to obtain the code, and then describe two methods to install the simulator. We explain how to run the experiments and plot the results. We also publicize  all the workload traces used in our paper.

## Organization

This repository is organized as follows.

- `algorithms` contains the implementation for Proteus as well as the baselines
- `configs` contains the configuration files used to drive the simulator. Each configuration file specifies a workload trace, the algorithms to use, and any relevant hyper-parameters
- `core` contains the simulator implementation files
- `figures` contains all the figures plotted by the scripts in the `plotting` directory
- `gurobi` contains the Gurobi licence used to run the optimization for Proteus. If you use obtain your own Gurobi license, it should be placed in this directory
- `logs` contains the logs gathered from the simulator for each experiment used to generate the figures
- `plotting` contains the scripts to plot results from the collected logs from the simulator
- `profiling` contains the profiled information of the model variants used by Proteus to drive its resource allocation algorithm
- `traces` contains all workload traces used in our paper
- `run.py` is the script used to start the simulator using a configuration file

## Experimentation

You can test Proteus using two methods as described below.

1. Using the Proteus Docker image with a pre-installed Gurobi trial license for quick testing

2. Installing Proteus locally using a `conda` environment and setting up a Gurobi license

We strongly recommend using Method 2 if you want to use Proteus or the simulator for your experiments. However, if you want to quickly test Proteus, we recommend Method 1.

## Method 1: Docker

Proteus uses the Gurobi optimization software to solve its MILP formulation. Although Gurobi requires a full-usage license, we have prepared a [Docker image](https://hub.docker.com/r/sohaibahmad759/proteus) with a pre-installed trial license to allow quick testing. However, we strongly recommend that you obtain a Gurobi license if you want to use Proteus.

Follow the instructions [here](DOCKER.md) to run Proteus using Docker.

## Method 2: Local installation

To use Proteus, we recommend obtaining a Gurobi license and creating a `conda` environment.

### Obtaining repo

You can clone the GitHub repo using:

`git clone https://github.com/UMass-LIDS/Proteus.git && cd Proteus`

### Conda environment setup

Once the repo is cloned, setup the `conda` environment as follows:

```bash
conda create --name proteus python=3.9.6
conda activate proteus`
python -m pip install -r requirements.txt
```

### Gurobi license

To complete the local installation, you need to obtain a Gurobi license as following.

1. Follow the instructions [here](https://www.gurobi.com/solutions/licensing/) to get a commercial or a free academic license for Gurobi, depending on your use.

2. Once you have obtained the license, Gurobi will provide a `gurobi.lic` file.

3. Place the license file under the path `gurobi/gurobi.lic`.

### Running experiments

Once you have followed the instructions above successfully, your local installation is complete. You can now follow the instructions [here](EXAMPLES.md) to run experiments with Proteus and the comparison baselines on various traces using the simulator.


## Citation

If you use this artifact or find it helpful, please cite our paper.

```bash
@article{ahmad2024proteus,
  title={Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling},
  author={Ahmad, Sohaib and Guan, Hui and Friedman, Brian D. and Williams, Thomas and Sitaraman, Ramesh K. and Woo, Thomas},
  booktitle={29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS ’24)},
  year={2024},
  doi={https://doi.org/10.1145/3617232.3624849}
}
```
