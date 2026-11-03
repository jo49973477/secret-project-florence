# Scripts

Run commands from the repository root unless a guide says otherwise.

| Task | Entry point |
| --- | --- |
| Florence A–D full-dataset experiments | `bash scripts/train_florence_ablations.sh A` (or `B`, `C`, `D`) |
| Explicit five-episode debug runs | `bash scripts/train_florence_ablations.sh overfit5_temporal_lora` (see `--help`) |
| Hydra training infrastructure | `train_hydra.py`, `train_hydra_worker.py`, `preflight_florence_ablations.py`, `verify_pretrained_dit.py` |
| UniVTAC Sidecar compatibility launcher | `train_univtac_sidecar.sh` / `train_univtac_sidecar.py` |
| Five-episode dataset workflow and checkpoint export | `univtac_overfit_workflow.py`, `export_zero3_checkpoint.py` |
| Server and rollout | `run_gr00t_univtac_server.sh`, `run_gr00t_univtac_rollout.sh` |

The SO100 and UniVTAC overfit baseline launchers remain at this level while their
callers and regression tests depend on them. `lib/checkpoint_resume.sh` supports
those launchers and the archived ones.

- [`diagnostics/`](diagnostics/README.md): verification, benchmarks, and sensor smoke tests.
- [`data/`](data/README.md): dataset downloads, repair, and conversion checks.
- [`legacy/`](legacy/README.md): historical launchers; use the Florence launcher for new A–D runs.
- [`deployment/`](deployment/README.md): platform setup, export, and inference tools.
- `lerobot_conversion/`: LeRobot format conversion tools.
- `eval/`: simulation evaluation readiness checks.
