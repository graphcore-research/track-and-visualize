# numerics vis

  

Repo for working on numerics-vis

  
  

## Installation

From the root directory of the repository, run:
```

python3 -m venv .venv

echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})/src:\$(dirname \${VIRTUAL_ENV})/experiments/Transformer:\$(dirname \${VIRTUAL_ENV})/experiments/Transformer/training" >> .venv/bin/activate

source .venv/bin/activate

pip install -r requirements.txt
