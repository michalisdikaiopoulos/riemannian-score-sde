# [Safe Sampling in Riemannian Score-Based Diffusion Models](https://drive.google.com/file/d/1ybcQfgWAu_wn-OxBztUW95DC-D39vNr8/view)

This repository builds on the [Riemannian Score-Based Generative Modeling](https://arxiv.org/abs/2202.02763) codebase and extends it by adding inference-time negative guidance for safe generation.

For the initial work on RSGMs, please advise this [repository](https://github.com/oxcsml/riemannian-score-sde)

## Install

Simple install instructions are:
```
git clone https://github.com/michalisdikaiopoulos/riemannian-score-sde.git
cd riemannian-score-sde
git clone https://github.com/oxcsml/geomstats.git
git -C geomstats apply ../patches/geomstats.patch
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_exps.txt
pip install -r geomstats/requirements.txt
GEOMSTATS_BACKEND=jax pip install -e geomstats --no-build-isolation --config-settings editable_mode=compat
pip install -e .
```

## Code structure

The safe-inference work sits on top of the base `score_sde` / `riemannian_score_sde` sampling code, and is spread across:

- `config/safety/*.yaml` — safety configuration: how the unsafe region is defined (`region: wedge` or `component`), which correction `method` is active, and its hyperparameters (`eta`, `beta_max`, `t_min`/`t_max`, `rejection_radius`, ...).
- `score_sde/models/flow.py` — the safety mechanisms themselves: `make_kernel_repulsion_score_fn` (the score-correction methods `early_window`, `full_window_scaled` and `late_window_scaled`, which wrap the learned score with a repulsion term near the unsafe region), and `sample_safe_noise` (the noise-rejection method, which biases the reverse SDE's starting points instead of correcting the score). Dispatch between them happens in `SDEPushForward.get_sampler`.
- `run.py` — collects unsafe reference points from the dataset (`_get_unsafe_mask_fn`, `_unsafe_wedge_mask`, `_unsafe_component_mask`), forward-diffuses them for the noise-rejection method, dispatches to the configured safety method, and computes the evaluation metrics: violation rate (VR, proportion of generated samples landing in the unsafe region) and MMD between generated and real samples.
- `riemannian_score_sde/datasets/{ring,kent_synthetic,checkerboard}.py` — synthetic datasets with controllable, known unsafe regions, used to test and visualize the safety mechanisms quickly (as opposed to the earth datasets, which are slower to train and less interpretable geometrically).
- `riemannian_score_sde/utils/vis.py` — plotting/animation for safety runs (safe vs. unsafe region overlays, score-magnitude/trajectory plots, `animate_sampling`).
- `inference_viz.ipynb` — interactive notebook for exploring a trained checkpoint's reverse diffusion trajectories with/without the safety mechanism applied, and sweeping methods against VR/MMD.

## Running experiments
Experiment configuration is handled by [hydra](https://hydra.cc/docs/intro/), a highly flexible `yaml` based configuration package. Base configs can be found in `config`, and parameters are overridden in the command line. Sweeps over parameters can also be managed with a single command.

Jobs scheduled on a cluster using a number of different plugins. We use Slurm, and configs for this can be found in `config/server` (note these are reasonably general but have some setup-specific parts). Other systems can easily be substituted by creating a new server configuration.

The main training and testing script can be found in `run.py`, and is dispatched by running `python main.py [OPTIONs]`.

### Training
To validate that the code is installed correctly and the RSGM models are training properly, run a toy experiment on the sphere:
```
python main.py experiment=s2_toy
```

To train a model to use with the safety mechanisms, run one of the synthetic experiments, e.g.:
```
python main.py experiment=ring_synthetic steps=100000
```
By default `mode=all`, so this both trains the model and runs a test-stage evaluation/plotting pass at the end. Checkpoints and logs land under `results/<experiment>/<override_dirname>/0`.

### Running inference without safety
`safety.*` overrides are excluded from the run directory name (see `config/server/base.yaml`), so you can re-run inference against an already-trained checkpoint by matching its training overrides and adding `mode=test resume=true`:
```
python main.py experiment=ring_synthetic steps=100000 mode=test resume=true safety.enabled=false
```
This restores the checkpoint and runs the test-stage evaluation/plotting with the safety mechanism off — the baseline sampling behaviour, with violation rate (VR) and MMD reported against the unsafe region without any correction applied.

### Adding safety at inference
Each method is selected with `safety.method=<name>` on top of `safety.enabled=true` (using `safety=default` for the wedge region, or `safety=kent_component` for a Kent mixture component), with the same `mode=test resume=true` re-run pattern as above.

**`early_window`** — the repulsion correction is applied for the whole reverse trajectory except very close to the end (disabled once `t` drops below `t_min`):
```
python main.py experiment=ring_synthetic steps=100000 mode=test resume=true \
    safety.enabled=true safety.method=early_window \
    safety.eta=5.0 safety.beta_max=10.0 safety.t_min=0.0 safety.n_max=5
```

**`full_window_scaled`** — the repulsion correction is applied across the entire trajectory (no time windowing), with the unsafe-direction and correction vectors rescaled to match the learned score's magnitude:
```
python main.py experiment=ring_synthetic steps=100000 mode=test resume=true \
    safety.enabled=true safety.method=full_window_scaled \
    safety.eta=5.0 safety.beta_max=10.0
```

**`late_window_scaled`** — the repulsion correction is only applied inside the `[t_min, t_max]` window (the learned score is left untouched elsewhere), with the same magnitude scaling as `full_window_scaled`:
```
python main.py experiment=ring_synthetic steps=100000 mode=test resume=true \
    safety.enabled=true safety.method=late_window_scaled \
    safety.eta=5.0 safety.beta_max=10.0 safety.t_min=0.0 safety.t_max=0.2
```
