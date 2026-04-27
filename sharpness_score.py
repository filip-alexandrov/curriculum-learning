import os
import pandas as pd

from autrainer.core.utils import Timer, set_device
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from aucurriculum.curricula.scoring import AbstractScore


class SharpnessScore(AbstractScore):
    """Per-sample sharpness scoring based on parameter-space loss landscape
    probes along random 1D directions. For each random direction, we normalize
    it per-filter to match the parameter norms, then sweep along the direction
    and compute the loss at each point. The score for a sample is defined as
    max(loss_line) - loss_center (higher means sharper).

    Notes:
        - Uses batched inference: for each step along the direction, the model
          parameters are set once and inference is run over the entire dataset
          in batches.
        - Multiple random directions can be averaged for more robust estimates.
    """

    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        run_name: str,
        stop: str = "best",
        subset: str = "train",
        num_steps: int = 11,
        step_range: tuple = (-1.0, 1.0),
        batch_size: int = 32,
        num_workers: int = 4,
        number_of_rand_directions: int = 1,
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
        self.num_steps = num_steps
        self.step_range = step_range
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.number_of_rand_directions = number_of_rand_directions

    def run(self, config: DictConfig, run_config: DictConfig, run_name: str) -> None:
        run_name, full_run_name = self.split_run_name(run_name)
        run_path = os.path.join(self.output_directory, full_run_name)
        data, model = self.prepare_data_and_model(run_config)
        dataset = self.get_dataset_subset(data, self.subset)
        
        device = set_device(config.device)
        
        # Setup DataLoader with optimizations matching autrainer training
        use_cuda = device.type == "cuda"
        pin_memory = use_cuda
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,
        )
        self.load_model_checkpoint(model, run_name)

        # criterion with reduction='none' to get per-sample losses
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # Collect labels once (need to iterate to get them)
        labels = []
        for _, y, _ in DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
            labels.extend(y.cpu().numpy().tolist())
        labels = np.array(labels, dtype=np.int64)
        num_samples = len(labels)

        # Store per-point losses for all random directions
        # all_pt_losses[direction_idx][pt_str] = np.array of shape (num_samples,)
        all_pt_losses = []

        # Loop over multiple random directions with progress bar
        direction_pbar = tqdm(
            range(self.number_of_rand_directions),
            desc="Random directions",
            disable=not config.get("progress_bar", False)
        )
        for direction_idx in direction_pbar:
            # Use seed = direction_idx + 1 for each random direction (seed=1, 2, 3, ...)
            current_seed = direction_idx + 1
            
            direction_pbar.set_description(
                f"Random directions [{direction_idx + 1}/{self.number_of_rand_directions}] (seed={current_seed})"
            )
            
            # Compute sharpness for this random direction
            pt_losses = self._compute_sharpness_for_direction(
                model, orig_weights=None, loader=loader, device=device, 
                criterion=criterion, config=config, pin_memory=pin_memory,
                seed=current_seed, num_samples=num_samples
            )
            
            all_pt_losses.append(pt_losses)
            
            # Save intermediate results for this seed
            seed_filename = os.path.join(run_path, f"sharpness_seed_{current_seed}.csv")
            self._save_grid_losses(pt_losses, labels, data, seed_filename)
        
        # Now aggregate all directions: average per-point losses across directions
        # Then compute final sharpness score
        aggregated_pt_losses = self._aggregate_grid_losses(all_pt_losses)
        
        # Compute final sharpness scores from aggregated grid losses
        final_scores = self._compute_final_scores(aggregated_pt_losses, num_samples)
        
        # Save final averaged scores
        df = self.create_dataframe(final_scores, labels, data)
        self.save_scores(df, run_path)
    
    def _compute_sharpness_for_direction(
        self, model, orig_weights, loader, device, criterion, config, pin_memory, 
        seed, num_samples
    ):
        """Compute line point losses for a single random direction.
        
        Returns:
            pt_losses: dict mapping step value (e.g., "-1.0", "0.0", "1.0") to numpy array of losses
        """
        # helpers for direction manipulation
        def normalize_directions_filter(direction, weights, eps=1e-10, ignore_biasbn=True):
            normed = []
            for d, w in zip(direction, weights):
                if d.dim() <= 1:
                    if ignore_biasbn:
                        normed.append(torch.zeros_like(d))
                    else:
                        normed.append(w.clone())
                else:
                    dcopy = d.clone()
                    first_dim = dcopy.size(0)
                    for i in range(first_dim):
                        df = dcopy[i]
                        wf = w[i]
                        denom = df.norm().clamp(min=eps)
                        factor = wf.norm() / denom
                        dcopy[i] = df * factor
                    normed.append(dcopy)
            return normed

        def apply_direction_to_model(m, orig_weights, dw, step):
            """Apply perturbation to model parameters along single direction.
            
            Args:
                m: model
                orig_weights: original model weights
                dw: direction (list of tensors)
                step: scalar step size along the direction
            """
            for p, w, d in zip(m.parameters(), orig_weights, dw):
                p.data.copy_(w + step * d)

        def restore_model_weights(m, orig_weights):
            """Restore original model parameters."""
            for p, w in zip(m.parameters(), orig_weights):
                p.data.copy_(w)

        def compute_all_losses_batched(m, loader, device, criterion, show_progress=False, non_blocking=False):
            """Run batched inference over entire dataset, return per-sample losses as tensor."""
            all_losses = []
            batch_iter = tqdm(
                loader,
                desc="  Batched inference",
                leave=False,
                disable=not show_progress
            )
            for x, y, _ in batch_iter:
                x = x.to(device, non_blocking=non_blocking)
                y = y.to(device, non_blocking=non_blocking)
                out = m(x)
                losses = criterion(out, y)  # per-sample losses
                all_losses.append(losses)
            return torch.cat(all_losses, dim=0)

        # Setup
        model.to(device)
        model.eval()
        
        # Save original weights once - keep on same device as model
        orig_weights = [p.data.clone() for p in model.parameters()]
        
        # Set seed for reproducible random direction
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Make single random direction and normalize
        dw = [torch.randn_like(p.data) for p in model.parameters()]
        dw = normalize_directions_filter(dw, orig_weights)

        # Create 1D line of step values (e.g., -1.0, -0.8, ..., 0.0, ..., 0.8, 1.0)
        steps = np.linspace(self.step_range[0], self.step_range[1], self.num_steps)
        
        # Find center index (step=0)
        center_idx = np.argmin(np.abs(steps))
        
        # Store losses for each step: dict[step_value] -> losses tensor
        loss_dict = {}

        # Batched evaluation: for each step along the direction
        show_progress = config.get("progress_bar", False)
        non_blocking = pin_memory
        progress_bar = tqdm(
            total=len(steps),
            desc=f"Sharpness scoring ({len(steps)} steps)",
            disable=not show_progress
        )
        
        with torch.no_grad():
            for step in steps:
                # Set model params for this step
                apply_direction_to_model(model, orig_weights, dw, float(step))
                
                # Compute losses for ALL samples at this step
                all_losses = compute_all_losses_batched(
                    model, loader, device, criterion,
                    show_progress=show_progress,
                    non_blocking=non_blocking
                )
                
                loss_dict[step] = all_losses  # Keep as tensor on device
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        # Restore original weights
        restore_model_weights(model, orig_weights)

        # Convert loss_dict to step-based format: step_str -> numpy array
        pt_losses = {}
        for step, loss_tensor in loss_dict.items():
            # Use step value as column name (formatted to avoid floating point issues)
            step_str = f"{step:.4f}"
            pt_losses[step_str] = loss_tensor.cpu().numpy()
        
        return pt_losses
    
    def _save_grid_losses(self, pt_losses, labels, data, filename):
        """Save per-point losses as CSV with columns: index, label, pt_0_0, pt_0_1, ..."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create index array (0 to num_samples-1)
        num_samples = len(labels)
        indices = np.arange(num_samples)
        
        # Build dataframe
        df_dict = {
            'index': indices,
            'label': labels,
        }
        
        # Add each grid point as a column
        for pt_str in sorted(pt_losses.keys()):
            df_dict[pt_str] = pt_losses[pt_str]
        
        df = pd.DataFrame(df_dict)
        df.to_csv(filename, index=False)
        print(f"  Saved grid losses to {filename}")
    
    def _aggregate_grid_losses(self, all_pt_losses):
        """Average grid point losses across all random directions.
        
        Args:
            all_pt_losses: List of dicts, each dict maps pt_str -> numpy array
        
        Returns:
            aggregated_pt_losses: dict mapping pt_str -> averaged numpy array
        """
        aggregated = {}
        
        # Get all point names from first direction
        point_names = sorted(all_pt_losses[0].keys())
        
        for pt_str in point_names:
            # Stack losses from all directions
            losses_list = [pt_losses[pt_str] for pt_losses in all_pt_losses]
            stacked = np.stack(losses_list, axis=0)  # shape: (num_directions, num_samples)
            
            # Average across directions
            aggregated[pt_str] = stacked.mean(axis=0)  # shape: (num_samples,)
        
        return aggregated
    
    def _compute_final_scores(self, aggregated_pt_losses, num_samples):
        """Compute final sharpness scores from aggregated line losses.
        
        Args:
            aggregated_pt_losses: dict mapping step_str -> numpy array of shape (num_samples,)
            num_samples: number of samples
        
        Returns:
            scores: numpy array of shape (num_samples,) with final sharpness scores
        """
        # Parse step values from column names
        step_values = []
        for step_str in aggregated_pt_losses.keys():
            step_values.append(float(step_str))
        
        # Find center (step closest to 0)
        center_step = min(step_values, key=lambda x: abs(x))
        center_step_str = f"{center_step:.4f}"
        loss_center = aggregated_pt_losses[center_step_str]
        
        # Find max loss for each sample across all non-center steps
        edge_steps = [s for s in step_values if s != center_step]
        all_edge_losses = np.stack(
            [aggregated_pt_losses[f"{s:.4f}"] for s in edge_steps], 
            axis=1
        )
        loss_max = all_edge_losses.max(axis=1)
        
        # Sharpness score per sample
        scores = ((loss_max - loss_center) / (1 + loss_center) * 100.0).astype(np.float32)
        
        return scores