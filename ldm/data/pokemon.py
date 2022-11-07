import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
from datasets import load_dataset
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self._prepare()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        pass
        # raise NotImplementedError()

    def _load(self):
        
        self.data = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

