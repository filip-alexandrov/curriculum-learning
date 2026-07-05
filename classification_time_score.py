import os
from typing import Tuple

import autrainer
from autrainer.datasets import AbstractDataset
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from aucurriculum.curricula.scoring import AbstractScore
from aucurriculum.curricula.scoring.utils import load_hydra_configuration


class ClassificationTimeScore(AbstractScore):
    def __init__(
        self,
        output_directory: str,
        results_dir: str,
        experiment_id: str,
        dataset: str,
        threshold: float = 0.5,
        subset: str = "train",
    ) -> None:
        """Scoring function based on human Minimum Viewing Time (MVT).

        Computes per-image difficulty as the minimum image presentation
        duration (in ms) at which humans correctly classify the image
        above ``threshold`` accuracy, using data from the MVT paper
        (NeurIPS 2023: "How hard are computer vision datasets?").

        Lower MVT → easier image; higher MVT → harder image.
        Images with no human responses receive the maximum duration (10000 ms).

        Args:
            output_directory: Directory where scores will be stored.
            results_dir: Directory where training results are stored.
            experiment_id: ID of the grid-search experiment.
            dataset: ID of the dataset config (e.g. ``"Difficulty-ImageNet"``).
            threshold: Minimum accuracy at a duration for that duration to
                count as a successful MVT. Defaults to 0.5.
            subset: Dataset subset to score in ["train", "dev", "test"].
                Defaults to "train".
        """
        super().__init__(
            output_directory=output_directory,
            results_dir=results_dir,
            experiment_id=experiment_id,
            run_name=None,
            subset=subset,
            reverse_score=False,
        )
        self.dataset_id = dataset
        self.threshold = threshold

    def preprocess(self) -> Tuple[list, list]:
        config = OmegaConf.create({})
        config.dataset = load_hydra_configuration("dataset", self.dataset_id)
        run_name = f"{config.dataset.id}_ClassificationTime"
        return [config], [run_name]

    def run(
        self,
        config: DictConfig,
        run_config: DictConfig,
        run_name: str,
    ) -> None:
        run_config.dataset.pop("criterion", None)
        run_config.dataset.pop("transform", None)
        data = autrainer.instantiate(
            config=run_config.dataset,
            instance_of=AbstractDataset,
            batch_size=1,
            seed=0,
        )

        df = self.get_dataframe(data, self.subset)

        human_csv = os.path.join(data.path, "human_responses.csv")
        human_df = pd.read_csv(human_csv)
        human_df["correct"] = (human_df["label"] == human_df["response"]).astype(int)

        accuracy_df = (
            human_df.groupby(["image", "image_duration"])["correct"]
            .mean()
            .reset_index()
        )
        accuracy_df.columns = ["image", "duration", "accuracy"]

        mvt_per_image = (
            accuracy_df.groupby("image")
            .apply(self._compute_mvt, include_groups=False)
            .reset_index()
        )
        mvt_per_image.columns = ["image", "mvt"]

        max_mvt = float(human_df["image_duration"].max())
        mvt_map = dict(zip(mvt_per_image["image"], mvt_per_image["mvt"]))

        # Match by bare filename (index_column stores synset/filename paths)
        filenames = df[data.index_column].apply(os.path.basename)
        df = df.copy()
        df["scores"] = filenames.map(mvt_map).fillna(max_mvt)
        df["decoded"] = df[data.target_column]
        df["encoded"] = df["decoded"].apply(data.target_transform.encode)

        self.save_scores(df, os.path.join(self.output_directory, run_name))

    def _compute_mvt(self, group: pd.DataFrame) -> float:
        above = group[group["accuracy"] >= self.threshold]
        if above.empty:
            return group["duration"].max()
        return float(above["duration"].min())
