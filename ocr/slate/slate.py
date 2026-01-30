import math

import gym
import torch
from torch import nn

from torch import Tensor

from ocr.slate.base import Base
from ocr.slate.common.utils import linear_warmup
from .slate_module import SLATE_Module


class SLATE(Base):
    def __init__(self, ocr_config: dict, env_config: dict, observation_space: gym.Space, preserve_slot_order=False) -> None:
        super(SLATE, self).__init__(ocr_config, env_config, observation_space)
        self._module = SLATE_Module(ocr_config, env_config)
        self.preserve_slot_order = preserve_slot_order
        super()._init_pooling_fields()

        # optimizer
        self._opt = torch.optim.Adam(
            [
                {
                    "params": self._module.get_dvae_params(),
                    "lr": self._config.learning.lr_dvae,
                },
                {
                    "params": self._module.get_sa_params(),
                    "lr": self._config.learning.lr_enc,
                },
                {
                    "params": self._module.get_tfdec_params(),
                    "lr": self._config.learning.lr_dec,
                },
            ]
        )

    def set_preserve_slot_order(self, preserve_slot_order):
        self.preserve_slot_order = preserve_slot_order

    def __call__(self, obs: Tensor, with_attns=False, with_masks=False, previous_slots=None) -> Tensor:
        if self.preserve_slot_order and previous_slots is None:
            previous_slots = self._module(obs, with_attns, with_masks, previous_slots=None)
        return self._module(obs, with_attns, with_masks, previous_slots=previous_slots)

    def get_loss(self, obs: Tensor, masks, with_rep=False, with_mse=False, prev_slots=None) -> dict:
        if with_rep:
            metrics, slots = self._module.get_loss(obs, masks, with_rep, with_mse, prev_slots=prev_slots)
        else:
            metrics = self._module.get_loss(obs, masks, with_rep, with_mse, prev_slots=prev_slots)
        metrics.update(
            {
                "lr_dvae": torch.Tensor([self._opt.param_groups[0]["lr"]]),
                "lr_enc": torch.Tensor([self._opt.param_groups[1]["lr"]]),
                "lr_dec": torch.Tensor([self._opt.param_groups[2]["lr"]]),
            }
        )
        return (metrics, slots) if with_rep else metrics

    def update(self, obs: Tensor, masks, step: int) -> dict:
        self._module.update_tau(step)
        # update lr
        lr_warmup_factor = linear_warmup(
            step, 0, 1, 0, self._config.learning.lr_warmup_steps
        )
        lr_decay_factor = math.exp(
            step / self._config.learning.lr_half_life * math.log(0.5)
        )
        lr_dvae = self._config.learning.lr_dvae
        lr_enc = lr_decay_factor * lr_warmup_factor * self._config.learning.lr_enc
        lr_dec = lr_decay_factor * lr_warmup_factor * self._config.learning.lr_dec
        self._opt.param_groups[0]["lr"] = lr_dvae
        self._opt.param_groups[1]["lr"] = lr_enc
        self._opt.param_groups[2]["lr"] = lr_dec
        # update params
        return super().update(obs, masks, step)

    def visualize(self, obs: Tensor, slots, with_attns=False, normalize_slots=False):
        attns = self._module.visualize(obs, slots, with_attns=with_attns, normalize_slots=normalize_slots)
        return attns


class SLATE_stub(nn.Module):
    def __init__(self, slot_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slot_shape = slot_shape

    def forward(self, obs: torch.Tensor, *args, **kwargs):
        return torch.ones(obs.size()[0], *self._slot_shape, dtype=torch.float32, device=obs.device)