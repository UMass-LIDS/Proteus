# Proteus
Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling

You can test Proteus using two methods as described below.

1. Using the Proteus Docker image with a pre-installed Gurobi trial license for quick testing

2. Installing Proteus locally using a `conda` environment and setting up a Gurobi license


## Method 1: Docker

Proteus uses the Gurobi optimization software to solve its MILP formulation. Although Gurobi requires a full-usage license, we have prepared a [Docker image](https://hub.docker.com/r/sohaibahmad759/proteus) with a pre-installed trial license to allow quick testing. However, we strongly recommend that you obtain a Gurobi license if you want to use Proteus.

## Method 2: Local installation

To use Proteus, we recommend obtaining a Gurobi license and creating a `conda` environment.

### Obtaining repo

You can clone the GitHub repo using:

`https://github.com/UMass-LIDS/Proteus.git`

### Conda environment setup

You can setup the `conda` environment as follows:

`conda create --name proteus python=3.9.6`

`conda activate proteus`

`python -m pip install requirements.txt`

### Gurobi license

1. You can follow the instructions [here](https://www.gurobi.com/solutions/licensing/) to get a commercial or a free academic license for Gurobi, depending on your use.

2. Once you have obtained the license, Gurobi will provide a `gurobi.lic` file.

3. Place the license file under the path `gurobi/gurobi.lic`.

### Running experiments

Once you have followed the instructions above successfully, your local installation is complete. You can now follow the instructions [here](EXAMPLES.md) to run experiments with Proteus and the comparison baselines on various traces using the simulator.
