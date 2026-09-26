"""Two-rank tiny-model runner for test_zero3_checkpoint_smoke.py."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import TrainingArguments

from gr00t.experiment.trainer import Gr00tTrainer


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        value = torch.tensor([float(index % 2), 1.0])
        return {"features": value, "labels": torch.tensor(float(index % 2))}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, features, labels=None):
        logits = self.linear(features).squeeze(-1)
        loss = torch.nn.functional.mse_loss(logits, labels.float())
        return {"loss": loss, "logits": logits}


def run(output: Path, ds_config: Path, resume: str | None = None):
    trainer = Gr00tTrainer(
        model=TinyModel(),
        args=TrainingArguments(
            output_dir=str(output),
            max_steps=3,
            save_steps=2,
            save_total_limit=3,
            logging_steps=1,
            report_to="none",
            deepspeed=str(ds_config),
            dataloader_num_workers=0,
            disable_tqdm=True,
        ),
        train_dataset=TinyDataset(),
    )
    trainer.train(resume_from_checkpoint=resume)
    if dist.get_rank() == 0:
        (output / "smoke_final_state.json").write_text(
            json.dumps({"global_step": trainer.state.global_step})
        )
    dist.barrier()


def main():
    output = Path(sys.argv[1])
    ds_config = Path(sys.argv[2])
    resume = sys.argv[3] if len(sys.argv) > 3 else None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(output, ds_config, resume)


if __name__ == "__main__":
    main()
