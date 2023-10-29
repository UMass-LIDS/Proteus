# Instructions to use Proteus with Docker

We have prepared a [Docker image](https://hub.docker.com/r/sohaibahmad759/proteus) to quickly test Proteus.

## Pulling image

Pull the Proteus Docker image using the following command:

`docker pull sohaibahmad759/proteus`

## Run Docker container

Run the Docker image using the following command:

`docker run -it sohaibahmad759/proteus`

This will run an interactive bash session on the Docker container. 

## Comparing Proteus and baselines (Figure 4)

Once inside the interactive session, we can run the simulation using Proteus as well as the baselines for the end-to-end evaluation on the Twitter trace (similar to Section 6.3). Use the following commands to do so:

```bash
python run.py --config_file configs/medium-normal_load/proteus.json
python run.py --config_file configs/medium-normal_load/clipper_highacc.json
python run.py --config_file configs/medium-normal_load/clipper_lowacc.json
python run.py --config_file configs/medium-normal_load/infaas.json
python run.py --config_file configs/medium-normal_load/sommelier.json
```

Please note that it can take some time to run the simulations end-to-end. For example, it can take over an hour to run only Proteus on the Twitter trace using an AWS `t2.micro` instance.

To plot the results, run the following inside the interactive Docker session:

```bash
# Produces Figure 4
python plotting/endtoend.py
```

This will generate a figure named `timeseries_together.pdf` in the `figures` folder, corresponding to Figure 4 in Section 6.2 of the paper. Note that the results may differ slightly due to differences between the cluster environment and the simulator.

## Bursty trace (Figure 5)

To recreate the results from Section 6.3 on the bursty trace, run the following commands:

```bash
python run.py --config_file configs/zipf_exponential_bursty/proteus.json
python run.py --config_file configs/zipf_exponential_bursty/clipper_lowacc.json
python run.py --config_file configs/zipf_exponential_bursty/clipper_highacc.json
python run.py --config_file configs/zipf_exponential_bursty/infaas.json
python run.py --config_file configs/zipf_exponential_bursty/sommelier.json
```

To plot the results, run:
```bash
# Produces Figure 5
python plotting/bursty.py
```

This will generate a figure named `bursty.pdf` in the `figures` folder, corresponding to Figure 4 in Section 6.2 of the paper.

## Copying results from Docker container

You can download the generated figures from the Docker container to your host machine using the following Docker commands:

```bash
docker cp <container_id>:/figures/timeseries_together.pdf <local_path>/timeseries_together.pdf
docker cp <container_id>:/figures/bursty.pdf <local_path>/bursty.pdf
```

Replace `<container_id>` with the Docker container ID (you can see this with `docker ps`), and `<local_path>` with the path on your local host machine where you want to copy the generated figure.
