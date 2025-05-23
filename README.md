# reelrecs
A project on building movie recommendation system using neural networks.

# Running
I was using Python 3.10, all libraries needed are specified in `requirements.txt`.
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
or if you prefer uv 
```bash
uv venv -p 3.10 && source .venv/bin/activate && uv pip install -r requirements.txt
```
Data should be loaded to `data` directory (for instance, KION dataset should be in `data/data_kion/*.csv`).