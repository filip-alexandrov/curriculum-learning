import os
from typing import Dict

import numpy as np
import torch
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader
from tqdm import tqdm

from autrainer.core.utils import set_device
from omegaconf import DictConfig

from aucurriculum.curricula.scoring import AbstractScore


class GradientSharpnessScore(AbstractScore):
    """Per-sample sharpness scoring based on greedy per-sample gradient
    ascent in an L2 ball of radius `k` around the optimized parameters
    (r = k), following the same (globally, not per-filter, normalized)
    perturbation direction as non-adaptive SAM:
    `epsilon(w) = k * grad(L)(w) / ||grad(L)(w)||_2`.

    Unlike `SharpnessScore` (which probes shared random directions), each
    sample follows its own loss gradient: at every step the ascent
    direction is recomputed from that sample's own gradient at its current
    position, globally L2-normalized (across all non-bias/BatchNorm
    parameters) and scaled to a fixed step size of `k / num_steps`. A
    sample stops as soon as a step fails to increase its loss, or after
    `num_steps` steps (i.e. at the ball's edge `k`), whichever comes
    first. The score is the relative loss increase between the ascent
    endpoint and the original (center) loss, matching the normalization
    used by `SharpnessScore`.

    Note this deliberately normalizes *globally* rather than per-filter
    (as `SharpnessScore` does for its random directions): a per-sample
    gradient can have a near-zero component for individual filters (e.g.
    a filter irrelevant to that sample's prediction), and rescaling such
    a near-zero direction up to match the filter's own weight norm blows
    up that filter's perturbation by orders of magnitude. A single global
    norm over the whole (masked) gradient vector does not have this
    failure mode.

    Since every sample walks its own trajectory, each sample effectively
    needs its own copy of the model parameters. This is vectorized via
    `torch.func` (`vmap` + `grad` + `functional_call`) instead of a Python
    loop, but memory scales with `batch_size * parameter_count`, so
    `batch_size` here is expected to be much smaller than a training batch
    size.
    """

    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        run_name: str,
        stop: str = "best",
        subset: str = "train",
        k: float = 0.25,
        num_steps: int = 3,
        batch_size: int = 16,
        num_workers: int = 4,
        ignore_biasbn: bool = True,
    ) -> None:
        super().__init__(
            output_directory=output_directory,
            results_dir=results_dir,
            experiment_id=experiment_id,
            run_name=run_name,
            stop=stop,
            subset=subset,
            reverse_score=False,  # higher sharpness -> harder
        )
        self.k = k
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_biasbn = ignore_biasbn

    def run(self, config: DictConfig, run_config: DictConfig, run_name: str) -> None:
        run_name, full_run_name = self.split_run_name(run_name)
        run_path = os.path.join(self.output_directory, full_run_name)
        data, model = self.prepare_data_and_model(run_config)
        dataset = self.get_dataset_subset(data, self.subset)

        device = set_device(config.device)
        model.to(device)
        self.load_model_checkpoint(model, run_name)
        model.eval()

        use_cuda = device.type == "cuda"
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_cuda,
            persistent_workers=self.num_workers > 0,
        )

        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        base_params = {n: p.detach() for n, p in model.named_parameters()}
        base_buffers = {n: b.detach() for n, b in model.named_buffers()}
        perturbable = {
            n: p.dim() > 1 or not self.ignore_biasbn
            for n, p in base_params.items()
        }
        step_size = self.k / self.num_steps

        def sample_loss(
            params: Dict[str, torch.Tensor],
            buffers: Dict[str, torch.Tensor],
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
            return criterion(out, y.unsqueeze(0))

        grad_fn = vmap(grad(sample_loss), in_dims=(0, None, 0, 0))
        loss_fn = vmap(sample_loss, in_dims=(0, None, 0, 0))

        show_progress = config.get("progress_bar", False)
        all_scores, all_labels, all_steps_taken = [], [], []

        for x, y, _ in tqdm(
            loader,
            desc="Gradient-ascent sharpness scoring",
            disable=not show_progress,
        ):
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            bsz = x.shape[0]

            current_params = {
                n: p.unsqueeze(0).expand(bsz, *p.shape).clone()
                for n, p in base_params.items()
            }

            with torch.no_grad():
                loss_center = loss_fn(current_params, base_buffers, x, y)
            current_loss = loss_center.clone()
            active = torch.ones(bsz, dtype=torch.bool, device=device)
            steps_taken = torch.zeros(bsz, dtype=torch.long, device=device)

            for _ in range(self.num_steps):
                if not bool(active.any()):
                    break

                grads = grad_fn(current_params, base_buffers, x, y)
                directions = self._normalize_direction_global(
                    grads, perturbable
                )
                candidate_params = {
                    name: current_params[name] + step_size * directions[name]
                    for name in current_params
                }

                with torch.no_grad():
                    candidate_loss = loss_fn(
                        candidate_params, base_buffers, x, y
                    )

                improved = active & (candidate_loss > current_loss)
                for name in current_params:
                    param_mask = improved.view(
                        bsz, *([1] * (current_params[name].dim() - 1))
                    )
                    current_params[name] = torch.where(
                        param_mask, candidate_params[name], current_params[name]
                    )
                current_loss = torch.where(improved, candidate_loss, current_loss)
                steps_taken += improved.long()
                active = improved

            scores = (
                (current_loss - loss_center) / (1 + loss_center) * 100.0
            )
            all_scores.append(scores.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            all_steps_taken.append(steps_taken.detach().cpu().numpy())

        scores = np.concatenate(all_scores).astype(np.float32)
        labels = np.concatenate(all_labels).astype(np.int64)
        steps_taken = np.concatenate(all_steps_taken)

        df = self.create_dataframe(scores, labels, data)
        df["ascent_steps_taken"] = steps_taken
        self.save_scores(df, run_path)

    @staticmethod
    def _normalize_direction_global(
        grads: Dict[str, torch.Tensor],
        perturbable: Dict[str, bool],
        eps: float = 1e-12,
    ) -> Dict[str, torch.Tensor]:
        """Normalize a batched per-sample gradient to unit L2 norm over the
        *whole* (masked) parameter vector, mirroring the global (non-
        adaptive) SAM perturbation `rho * grad(L) / ||grad(L)||_2`, but
        computed per-sample rather than on the average batch loss.

        Args:
            grads: Per-sample gradient, one tensor of shape
                (batch, *param_shape) per parameter name.
            perturbable: Whether each named parameter participates in the
                direction (False zeroes bias / BatchNorm entries out,
                excluding them from both the norm and the step).

        Returns:
            Dict of unit-(global-)norm directions, same shapes as `grads`.
        """
        masked = {
            name: (g if perturbable[name] else torch.zeros_like(g))
            for name, g in grads.items()
        }
        batch = next(iter(masked.values())).shape[0]
        sq_sum = torch.zeros(batch, device=next(iter(masked.values())).device)
        for g in masked.values():
            sq_sum = sq_sum + g.flatten(1).pow(2).sum(dim=1)
        total_norm = sq_sum.sqrt().clamp(min=eps)  # (batch,)

        directions = {}
        for name, g in masked.items():
            view_shape = (batch,) + (1,) * (g.dim() - 1)
            directions[name] = g / total_norm.view(view_shape)
        return directions
