import os 
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import numpy as np 
import pandas as pd 
import torch 
from autrainer.datasets import BaseClassificationDataset
from autrainer.transforms import SmartCompose
from omegaconf import DictConfig
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor


class CIFAR100Wrapper(datasets.CIFAR100): 
    def __init__(self, transform: SmartCompose = None, *args, **kwargs) -> None: 
        super().__init__(*args, **kwargs)
        self._transform = transform
        self._convert_pil = ToTensor()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        img, target = super().__getitem__(index)
        img = (self._convert_pil(img) * 255).to(torch.uint8) # convert back to uint8

        if self._transform: 
            img = self._transform(img, index=index)

        return img, target, index
    
class CIFAR100(BaseClassificationDataset):
    def __init__(
            self, 
            path: str, 
            seed: int, 
            metrics: List[Union[str, DictConfig, Dict]], 
            tracking_metric: Union[str, DictConfig, Dict],
            index_column: str, 
            target_column: str, 
            batch_size: int, 
            inference_batch_size: Optional[int] = None,
            train_transform: Optional[SmartCompose] = None, 
            dev_transform: Optional[SmartCompose] = None, 
            test_transform: Optional[SmartCompose] = None, 
            stratify: Optional[List[str]] = None, 
            dev_split: float = 0.0, 
            dev_split_seed: Optional[int] = None, 
    ) -> None: 
        """CIFAR-100 dataset.

        Args:
            path: Root path to the dataset.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
            dev_split: Fraction of the training set to use as the development
                set. Defaults to 0.0.
            dev_split_seed: Seed for the development split. If None, seed is
                used. Defaults to None.
        """
        self._assert_dev_split(dev_split) 
        self.dev_split = dev_split
        self.dev_split_seed = dev_split_seed or seed 
        super().__init__(
            path=path, 
                features_subdir="", 
                seed=seed, 
                metrics = metrics, 
                tracking_metric=tracking_metric, 
                index_column=index_column, 
                target_column=target_column,
                file_type="", 
                file_handler="autrainer.datasets.utils.IdentityFileHandler", 
                batch_size=batch_size, 
                inference_batch_size=inference_batch_size, 
                train_transform=train_transform, 
                dev_transform=dev_transform, 
                test_transform=test_transform, 
                stratify=stratify
                )
        
    def load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.dev_split == 0: 
            return ( 
                pd.read_csv(os.path.join(self.path, "train.csv")), 
                pd.read_csv(os.path.join(self.path, "test.csv")), 
                pd.read_csv(os.path.join(self.path, "test.csv"))
            )
        
        train_df = pd.read_csv(os.path.join(self.path, "train.csv"))
        indices = train_df[self.index_column].values.copy()
        rng = np.random.default_rng(self.dev_split_seed) 
        rng.shuffle(indices) 
        self.train_indices = indices[: int(len(indices) * (1 - self.dev_split))]
        self.dev_indices = indices[int(len(indices) * (1 - self.dev_split)) :]
        self.train_indices.sort()
        self.dev_indices.sort()

        return (
            train_df[train_df[self.index_column].isna(self.train_indices)].copy(), 
            train_df[train_df[self.index_column].isna(self.train_indices)].copy(), 
            pd.read_csv(os.path.join(self.path, "test.csv"))
        )
    
    @cached_property
    def train_dataset(self) -> CIFAR100Wrapper: 
        _train_dataset = CIFAR100Wrapper(
            root=self.path, 
            transform=self.train_transform, 
            train=True, 
            download = False
        )

        if self.dev_split == 0: 
            return _train_dataset
        
        return Subset(_train_dataset, self.train_indices) 
    
    @cached_property
    def dev_dataset(self) -> CIFAR100Wrapper: 
        if self.dev_split == 0: 
            return self.test_dataset
        
        return Subset(
            CIFAR100Wrapper(root= self.path, transform=self.dev_transform, train=False, download = False), 
            self.dev_indices
        )
    
    @cached_property
    def test_dataset(self) -> CIFAR100Wrapper: 
        return CIFAR100Wrapper(
            root=self.path, 
            transform=self.test_transform, 
            train=False, 
            download=False
        )
    
    @staticmethod
    def download(path: str) -> None:
        """Download the CIFAR-100 dataset.

        As torchvision.datasets.CIFAR100 automatically handles the dataset
        creation and no preprocessing is intended for image datasets at
        the moment, this method only downloads the dataset and prepares the CSV
        files with the labels.

        Args:
            path: Path to the directory to download the dataset to.
        """
        if os.path.isdir(os.path.join(path, "cifar-100-python")):
            return

        datasets.CIFAR100(root=path, download=True)
        train_dataset = datasets.CIFAR100(root=path, train=True, download=False)
        test_dataset = datasets.CIFAR100(root=path, train=False, download=False)
        train_df = pd.DataFrame(
            {"index": list(range(len(train_dataset))), "label": train_dataset.targets}
        )
        test_df = pd.DataFrame(
            {"index": list(range(len(test_dataset))), "label": test_dataset.targets}
        )
        train_df["label"] = train_df["label"].apply(lambda x: train_dataset.classes[x])
        test_df["label"] = test_df["label"].apply(lambda x: test_dataset.classes[x])

        train_df.to_csv(os.path.join(path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(path, "test.csv"), index=False)
