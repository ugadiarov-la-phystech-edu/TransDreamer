from collections import namedtuple

import torch
from omegaconf import OmegaConf

from ocr.slate.slate import SLATE
from ocr.tools import SlotExtractor


class SLATEExtractor(SLATE, SlotExtractor):
    def __init__(self, config_path, checkpoint_path, image_size, device):
        config_ocr = OmegaConf.load(config_path)
        config_env = namedtuple('EnvConfig', ['obs_size', 'obs_channels'])(image_size, 3)
        super().__init__(config_ocr, config_env, observation_space=None, preserve_slot_order=True)
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._image_size = image_size
        self._device = device
        self._opt = None
        self._config_ocr = config_ocr

        state_dict = torch.load(checkpoint_path, weights_only=False, map_location='cpu')["ocr_module_state_dict"]
        self._module.load_state_dict(state_dict)
        self.eval()
        self.requires_grad_(False)
        self._module.to(self._device)

    @property
    def n_slots(self):
        return self._config_ocr.slotattr.num_slots

    @property
    def dim(self):
        return self._config_ocr.slotattr.slot_size
