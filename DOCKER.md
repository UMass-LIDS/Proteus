# Instructions to use Proteus with Docker

We have prepared a [Docker image](https://hub.docker.com/r/sohaibahmad759/proteus) to quickly test Proteus.

## Pulling image

Pull the Proteus Docker image using the following command:

`docker pull sohaibahmad759/proteus`

## Run Docker container

Run the Docker image using the following command:

`docker run -it sohaibahmad759/proteus`

This will run an interactive bash session on the Docker container. 

## Running Proteus and baselines

Once inside the interactive session, we can run the simulation using Proteus as well as the baselines. Use the following commands to do so:

```bash
python run.py --config_file configs/zipf_exponential/proteus.json
python run.py --config_file configs/zipf_exponential/clipper_highacc.json
python run.py --config_file configs/zipf_exponential/infaas.json
python run.py --config_file configs/zipf_exponential/sommelier.json
```

Please note that it can take some time to run the simulations end-to-end. For example, it can take over an hour to run Proteus on the Twitter trace using an AWS `t2.micro` instance.

## Plotting results

To plot results, run the following inside the interactive Docker session:

`python plotting/endtoend.py`

This will generate a figure in the `figures` folder. 

## Copying results from Docker container

You can download the generated figures from the Docker container to your host machine using the following Docker command:

`docker cp <container_id>:/figures/timeseries_together.pdf <local_path>/timeseries_together.pdf`

Replace `<container_id>` with the Docker container ID (you can see this with `docker ps`), and `<local_path>` with the path on your local host machine where you want to copy the generated figure.

## Bursty trace

To recreate the results from Section 6.4 on the bursty trace, run the following commands:

```bash
python run.py --config_file configs/zipf_exponential_bursty/proteus.json
python run.py --config_file configs/zipf_exponential_bursty/clipper_lowacc.json
python run.py --config_file configs/zipf_exponential_bursty/infaas.json
python run.py --config_file configs/zipf_exponential_bursty/sommelier.json
python plotting/bursty.py
```
