# Baseline Runners

This folder holds local baseline setup and timing helpers.

## DetIE on CaRB dev

Use one of the bash setup scripts on the A100 server:

```bash
# From the DiffIE repo root.
bash baselines/setup_detie_local.sh
```

That path uses micromamba under `baselines/.micromamba`, with no sudo. If you
prefer a plain `venv` + pip path, use:

```bash
bash baselines/setup_detie_pip.sh
```

DetIE does not use `uv`; upstream dependency pins live in
`baselines/DetIE/context/requirements.txt`.

Then download the DetIE files bundle from the upstream README:

```text
https://drive.google.com/drive/folders/1SGeQWcFwmL4BaMbCTxVw5-oU69vPW_d-?usp=sharing
```

Place the LSOIE checkpoint folder here:

```text
baselines/DetIE/results/logs/default/version_243/
```

The timing script checks for:

```text
baselines/DetIE/results/logs/default/version_243/checkpoints/best.ckpt
baselines/DetIE/results/logs/default/version_243/hparams.yaml
```

Run timing:

```bash
# Activate the environment printed by setup_detie_local.sh first.
bash baselines/time_detie_carb_dev.sh
```

It writes:

```text
baselines/detie_carb_dev_timing.csv
baselines/detie_carb_dev_timing_logs/
```
