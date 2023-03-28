import numpy as np

from pathlib import Path
from data.base import BaseMMSeg
from data import utils
from config import dataset_dir


STREETHAZARDS_CONFIG_PATH = Path(__file__).parent / "config" / "streethazards.py"
STREETHAZARDS_CATS_PATH = Path(__file__).parent / "config" / "streethazards.yml"

class StreetHazardsDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(image_size, crop_size, split, STREETHAZARDS_CONFIG_PATH, **kwargs)
        self.names, self.colors = utils.dataset_cat_description(STREETHAZARDS_CATS_PATH)
        self.n_cls = 14
        self.ignore_label = 0
        self.reduce_zero_label = False

    def update_default_config(self, config):

        root_dir = dataset_dir()
        path = Path(root_dir) / "streethazards"
        config.data_root = path

        config.data[self.split]["data_root"] = path
        config = super().update_default_config(config)
        

        return config

    def test_post_process(self, labels):
        return labels