# Emergency SAGIN connectivity for Qusin (Nablus): Q-learning vs. fair baselines

Code and results for the tutorial article *Evaluating Reinforcement Learning for Emergency SAGIN Connectivity:
A Tutorial Case Study of Qusin, Nablus* (Mohammed Allan Yadak, An-Najah National University).

These scripts are built **on top of** the open-source toolkit
[Lewis-panda/Optimizing-SAGIN-Association-Link-Power-Using-Reinforcement-Learning](https://github.com/Lewis-panda/Optimizing-SAGIN-Association-Link-Power-Using-Reinforcement-Learning).
The toolkit's own files are **not** included here and are **not modified**.

## Setup (Windows cmd; use `python3` on Linux/macOS)
```
git clone https://github.com/Lewis-panda/Optimizing-SAGIN-Association-Link-Power-Using-Reinforcement-Learning.git sagin
git clone <URL of this repository> mycode
copy mycode\*.py sagin\
xcopy mycode\results_qusin sagin\results_qusin\ /E /I
xcopy mycode\results_qusin_v2 sagin\results_qusin_v2\ /E /I
cd sagin
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python tests.py
```

## Run
```
python sagin_v2.py            # improved model, main run + sweeps (10-15 min)
python paired_stats.py        # 30 topologies, paired bootstrap CIs (~10 min)
python make_figures_v2.py     # PNG figures  -> results_qusin_v2/figures
python qusin_scenario.py      # first (flawed) model (~5 min)
python make_figures_v1.py     # figures of the first model
```
`python sagin_v2.py --quick` is a short smoke test. `make_ieee_figures.py` writes the vector PDF figures used in the paper
to `../1_Report/Figures` (set the environment variable `IEEE_FIG_DIR` to change the folder).

## Files
| File | Role |
|---|---|
| `qusin_scenario.py` | six villages (coordinates and 2017 populations), clustered user layout, first experiment |
| `sagin_v2.py` | improved model (spot beams, frequency reuse, capacity, directional backhaul), Q-learning, baselines, runner |
| `paired_stats.py` | 30-topology comparison with 95% bootstrap confidence intervals |
| `make_figures_v1.py`, `make_figures_v2.py`, `make_ieee_figures.py` | figures |
| `results_qusin/`, `results_qusin_v2/` | results used in the paper |

Seeds are fixed, so reruns give identical numbers on the same library versions. No GPU is needed.
Village data: public Wikipedia pages citing the PCBS 2017 census. AI assistance (Claude, Anthropic) was used for code and text drafting.
