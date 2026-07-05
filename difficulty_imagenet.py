import os
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision
from autrainer.datasets import BaseClassificationDataset
from autrainer.datasets.utils import AbstractFileHandler
from autrainer.transforms import SmartCompose
from omegaconf import DictConfig


class RGBImageFileHandler(AbstractFileHandler):
    def __init__(self) -> None:
        """Image file handler that loads images as uint8 RGB tensors.

        Handles RGBA (4-channel) images by dropping the alpha channel,
        and grayscale (1-channel) images by replicating to 3 channels.
        """

    def load(self, file: str) -> torch.Tensor:
        img = torchvision.io.read_image(file)
        if img.shape[0] == 4:
            img = img[:3]
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img

    def save(self, file: str, data: torch.Tensor) -> None:
        if data.dtype == torch.uint8:
            data = data / 255
        torchvision.utils.save_image(data, file)


class DifficultyImageNetWrapper(torch.utils.data.Dataset):
    """Internal PyTorch Dataset for DifficultyImageNet.

    Loads images from ``root/synset/filename`` paths listed in ``df``,
    applying the given transforms and target transform.
    """

    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        index_column: str,
        target_column: str,
        transform: Optional[SmartCompose] = None,
        target_transform=None,
    ) -> None:
        self.root = root
        self.df = df.reset_index(drop=True)
        self.index_column = index_column
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self._handler = RGBImageFileHandler()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int, int]:
        row = self.df.iloc[item]
        img_path = os.path.join(self.root, row[self.index_column])
        data = self._handler.load(img_path)
        target = row[self.target_column]

        if self.transform is not None:
            data = self.transform(data, index=item).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, item


class DifficultyImageNet(BaseClassificationDataset):
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
    ) -> None:
        """Difficulty ImageNet dataset from the MVT paper.

        Images are organized under ``cropped_images/<synset_id>/<filename>``.
        CSV splits (train/dev/test) are expected at ``path/train.csv``,
        ``path/dev.csv``, and ``path/test.csv`` with columns ``filename``
        (``synset_id/image_file``) and ``label``.

        Run :meth:`setup` to generate the CSV splits from the raw data.

        Args:
            path: Root path to the dataset (containing ``cropped_images/``
                and ``human_responses.csv``).
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Column containing image paths relative to
                ``cropped_images/``.
            target_column: Column containing class labels.
            batch_size: Batch size for data loaders.
            inference_batch_size: Batch size for inference.
            train_transform: Transform for the training set.
            dev_transform: Transform for the development set.
            test_transform: Transform for the test set.
            stratify: Columns to stratify splits on.
        """
        super().__init__(
            path=path,
            features_subdir="cropped_images",
            seed=seed,
            metrics=metrics,
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
            stratify=stratify,
        )

    @cached_property
    def train_dataset(self) -> DifficultyImageNetWrapper:
        return DifficultyImageNetWrapper(
            root=os.path.join(self.path, "cropped_images"),
            df=self.df_train,
            index_column=self.index_column,
            target_column=self.target_column,
            transform=self.train_transform,
            target_transform=self.target_transform,
        )

    @cached_property
    def dev_dataset(self) -> DifficultyImageNetWrapper:
        return DifficultyImageNetWrapper(
            root=os.path.join(self.path, "cropped_images"),
            df=self.df_dev,
            index_column=self.index_column,
            target_column=self.target_column,
            transform=self.dev_transform,
            target_transform=self.target_transform,
        )

    @cached_property
    def test_dataset(self) -> DifficultyImageNetWrapper:
        return DifficultyImageNetWrapper(
            root=os.path.join(self.path, "cropped_images"),
            df=self.df_test,
            index_column=self.index_column,
            target_column=self.target_column,
            transform=self.test_transform,
            target_transform=self.target_transform,
        )

    @staticmethod
    def setup(path: str, seed: int = 42) -> None:
        """Create stratified train/dev/test CSV splits (80/10/10 per class).

        Reads the image directory structure and ``human_responses.csv`` to
        build synset-to-label mappings, then writes ``train.csv``,
        ``dev.csv``, and ``test.csv`` to ``path``.

        Args:
            path: Root path to the dataset directory.
            seed: Random seed for the stratified split.
        """
        img_dir = os.path.join(path, "cropped_images")
        human_csv = os.path.join(path, "human_responses.csv")

        df_human = pd.read_csv(human_csv)

        records = []
        for synset in sorted(os.listdir(img_dir)):
            synset_path = os.path.join(img_dir, synset)
            if not os.path.isdir(synset_path) or synset.startswith("."):
                continue
            for img in sorted(os.listdir(synset_path)):
                if img.startswith("."):
                    continue
                records.append(
                    {
                        "filename": f"{synset}/{img}",
                        "synset": synset,
                        "image": img,
                    }
                )

        all_df = pd.DataFrame(records)

        synset_label_map = {}
        for synset, group in all_df.groupby("synset"):
            matching = df_human[df_human["image"].isin(group["image"])]
            if len(matching) > 0:
                synset_label_map[synset] = matching["label"].mode()[0]

        all_df["label"] = all_df["synset"].map(synset_label_map)
        all_df = all_df.dropna(subset=["label"])

        rng = np.random.default_rng(seed)
        train_rows, dev_rows, test_rows = [], [], []

        for label in sorted(all_df["label"].unique()):
            label_df = all_df[all_df["label"] == label].reset_index(drop=True)
            indices = np.arange(len(label_df))
            rng.shuffle(indices)
            n_train = int(len(indices) * 0.8)
            n_dev = int(len(indices) * 0.1)

            train_rows.append(label_df.iloc[indices[:n_train]])
            dev_rows.append(label_df.iloc[indices[n_train : n_train + n_dev]])
            test_rows.append(label_df.iloc[indices[n_train + n_dev :]])

        for split_name, rows in [
            ("train", train_rows),
            ("dev", dev_rows),
            ("test", test_rows),
        ]:
            split_df = pd.concat(rows)[["filename", "label"]].reset_index(drop=True)
            split_df.to_csv(os.path.join(path, f"{split_name}.csv"), index=False)
