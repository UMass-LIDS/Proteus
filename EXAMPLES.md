
# Examples

## Pre-requisities

Running these examples requires you to have a local installation of Proteus as well as a Gurobi license. Make sure to follow the instructions [here](README.md) to do so.

## Traces

We have provided all trace files used in the Proteus paper under `traces`.

## Configs

Running the simulator requires using a `config_file`. You can see examples of these in the `configs` folder. We have provided config files for the Twitter trace (`zipf_exponential`) as well as the bursty trace from Section 6.3 of the Proteus paper (`zipf_exponential_bursty`). However, you can modify these files to use any of the other trace files, such as `medium-normal_load` used for Section 6.2.

## End-to-end comparison

To run an end-to-end comparison of Proteus with the baselines, run the following commands in order:

`python run.py --config_file configs/zipf_exponential/proteus.json`

`python run.py --config_file configs/zipf_exponential/clipper_lowacc.json`

`python run.py --config_file configs/zipf_exponential/infaas.json`

`python run.py --config_file configs/zipf_exponential/sommelier.json`

`python plotting/endtoend.py`

Please note that it can take some time to run the simulations end-to-end. For example, it can take over an hour to run Proteus on the Twitter trace using an AWS `t2.micro` instance.

## Bursty trace experiment

To recreate the results from Section 6.4 on the bursty trace, run the following commands in order:

`python run.py --config_file configs/zipf_exponential_bursty/proteus.json`

`python run.py --config_file configs/zipf_exponential_bursty/clipper_lowacc.json`

`python run.py --config_file configs/zipf_exponential_bursty/infaas.json`

`python run.py --config_file configs/zipf_exponential_bursty/sommelier.json`

`python plotting/bursty.py`
