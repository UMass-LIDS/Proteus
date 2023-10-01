# Proteus
Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling


`conda create --name proteus python=3.10.6`

`conda activate proteus`

`python -m pip install requirements.txt`

`python run.py --config_file configs/zipf_exponential/proteus.json`

`python run.py --config_file configs/zipf_exponential/clipper_lowacc.json`

`python run.py --config_file configs/zipf_exponential/infaas.json`

`python run.py --config_file configs/zipf_exponential/sommelier.json`

`python plotting/endtoend.py`
