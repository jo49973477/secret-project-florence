"""Regression checks for deterministic memorization shell defaults."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
OVERFIT_SCRIPTS = (
    ROOT / "scripts/overfit_so100_5eps.sh",
    ROOT / "scripts/overfit_univtac_5eps.sh",
    ROOT / "scripts/overfit_univtac_5eps_lora.sh",
)


@pytest.mark.parametrize("script", OVERFIT_SCRIPTS)
def test_overfit_scripts_disable_model_and_data_regularization(script):
    text = script.read_text()
    for required in (
        "--action-head-dropout 0",
        "--vl-self-attention-dropout 0",
        "--disable-color-jitter",
        "--random-rotation-angle 0",
        "--shortest-image-edge 256",
        "--crop-fraction 1.0",
        'checkpoint_resume_require_fresh_output "${OUTPUT_DIR}"',
    ):
        assert required in text


def test_so100_defaults_to_overfit_but_retains_official_mode():
    text = OVERFIT_SCRIPTS[0].read_text()
    assert 'MODE="${MODE:-overfit}"' in text
    assert "official)" in text


def test_univtac_lora_uses_single_step_accumulation_by_default():
    text = OVERFIT_SCRIPTS[2].read_text()
    assert 'GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"' in text
