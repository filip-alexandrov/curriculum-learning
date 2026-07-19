"""Evaluate trained WideResNet-28-4/-10 checkpoints under weight
quantization at a sweep of bit-widths, on the validation (dev) set.

Reproduces the exact metric computation used during normal autrainer
training/evaluation (see autrainer/autrainer/training/training.py,
`ModularTaskTrainer._evaluate`), so results at bits=32 (fp32, unquantized)
should reproduce the original `_best/dev.yaml` numbers as a sanity check.

Usage (run inside the cluster's aucurriculum/autrainer environment, from the
curriculum-learning/ directory, i.e. cluster_project/ once synced):

    python evaluate_quantized.py \\
        --config conf/quantization_models.yaml \\
        --output quantization_results.csv

See conf/quantization_models.yaml for the list of models/checkpoints and the
bit-width sweep, and quantization.py for the quantization scheme itself.
"""

import argparse
import os
import time
from typing import Any, Dict, List

import autrainer
from autrainer.core.utils import set_device
from autrainer.metrics import F1, UAR, Accuracy
from aucurriculum.curricula.scoring.abstract_score import AbstractScore
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from quantization import quantize_model_, quantization_error


def load_run_config(results_dir: str, experiment_id: str, run_name: str) -> DictConfig:
    cfg_path = os.path.join(
        results_dir, experiment_id, "training", run_name, ".hydra", "config.yaml"
    )
    return OmegaConf.load(cfg_path)


def load_checkpoint(results_dir: str, experiment_id: str, run_name: str) -> dict:
    ckpt_path = os.path.join(
        results_dir, experiment_id, "training", run_name, "_best", "model.pt"
    )
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Faithful reimplementation of the installed autrainer 0.4.0
    `ModularTaskTrainer._evaluate` (autrainer/training/training.py) for a
    standalone (non-training) checkpoint evaluation. Batches are plain
    (features, target, sample_idx) tuples in this version (no DataBatch
    class); the model is called with a single positional tensor
    (no `create_model_inputs`/`model.inputs` in this version); the loss is a
    running sum of per-batch scalar (mean-reduced) losses divided by
    `len(loader) + 1` -- an off-by-one quirk in the original trainer that is
    replicated here on purpose so fp32 results match the original
    `_best/dev.yaml` exactly.
    """
    model.eval()
    model.to(device)

    probabilities_fn = data.target_transform.probabilities_training
    outputs, targets = [], []
    loss_sum = 0.0
    for features, target, sample_idx in loader:
        output = model(features.to(device))
        loss_sum += criterion(probabilities_fn(output), target.to(device)).cpu().item()
        outputs.append(output.cpu())
        targets.append(target.cpu())
    loss = loss_sum / (len(loader) + 1)

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0).numpy()
    probs = data.target_transform.probabilities_inference(outputs)
    preds = data.target_transform.predict_inference(probs)
    preds = preds if isinstance(preds, list) else [preds]

    return {
        "accuracy": Accuracy()(targets, preds),
        "f1": F1()(targets, preds),
        "uar": UAR()(targets, preds),
        "loss": loss,
    }


def run_one_model(
    entry: Dict[str, Any],
    results_dir: str,
    experiment_id: str,
    bits_sweep: List[int],
    per_channel: bool,
    keep_first_last_fp32: bool,
) -> List[Dict[str, Any]]:
    run_name = entry["run_name"]
    print(f"\n=== {entry['curriculum']} (k={entry['k']}) :: {run_name} ===", flush=True)

    run_config = load_run_config(results_dir, experiment_id, run_name)
    data, model = AbstractScore.prepare_data_and_model(run_config)
    device = set_device(run_config.device)

    # dev_loader is a @cached_property; batch_size comes from the
    # dataset's own inference_batch_size (set at construction time in
    # AbstractScore.prepare_data_and_model via run_config.batch_size /
    # .inference_batch_size), not passed here.
    loader = data.dev_loader
    criterion = autrainer.instantiate_shorthand(
        config=run_config.criterion,
        instance_of=torch.nn.modules.loss._Loss,
    )
    criterion.to(device)

    pristine_state_dict = load_checkpoint(results_dir, experiment_id, run_name)

    rows = []
    for bits in bits_sweep:
        model.load_state_dict(pristine_state_dict)
        quantize_model_(
            model,
            bits=bits,
            per_channel=per_channel,
            keep_first_last_fp32=keep_first_last_fp32,
        )
        qerr = quantization_error(pristine_state_dict, bits) if bits < 32 else 0.0

        t0 = time.time()
        metrics = evaluate(model, data, loader, criterion, device)
        dt = time.time() - t0

        row = {
            "curriculum": entry["curriculum"],
            "k": entry["k"],
            "run_name": run_name,
            "bits": bits,
            "mean_relative_weight_error": qerr,
            **metrics,
            "eval_seconds": dt,
        }
        rows.append(row)
        print(
            f"  bits={bits:>2} | acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} uar={metrics['uar']:.4f} "
            f"loss={metrics['loss']:.4f} | rel_w_err={qerr:.4f} | {dt:.1f}s",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="conf/quantization_models.yaml",
        help="YAML listing models/checkpoints and the bit-width sweep.",
    )
    parser.add_argument(
        "--output", default="quantization_results.csv",
        help="Where to write the combined results CSV.",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated substrings to filter run_names (for quick reruns/debugging).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    bits_sweep = list(cfg.bits)
    models = list(cfg.models)
    if args.only:
        needles = args.only.split(",")
        models = [m for m in models if any(n in m["run_name"] for n in needles)]

    all_rows = []
    for entry in models:
        rows = run_one_model(
            entry=entry,
            results_dir=cfg.results_dir,
            experiment_id=cfg.experiment_id,
            bits_sweep=bits_sweep,
            per_channel=cfg.per_channel,
            keep_first_last_fp32=cfg.keep_first_last_fp32,
        )
        all_rows.extend(rows)
        # Write incrementally so a partial run on the cluster still leaves
        # usable results if interrupted/pre-empted.
        pd.DataFrame(all_rows).to_csv(args.output, index=False)

    print(f"\nSaved {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
