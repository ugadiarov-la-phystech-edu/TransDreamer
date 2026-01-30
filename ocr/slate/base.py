import gym
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from torch import Tensor

from ocr.slate.common.utils import optimizer_to


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Base(BaseFeaturesExtractor):
    def __init__(self, ocr_config, env_config, observation_space: gym.Space) -> None:
        slotattr = ocr_config.slotattr
        super().__init__(observation_space, features_dim=slotattr.slot_size * slotattr.num_slots)
        self.name = ocr_config.name
        self._config = ocr_config
        self._obs_size = env_config.obs_size
        self._obs_channels = env_config.obs_channels

    def _init_pooling_fields(self):
        # for pooling layer
        self.rep_dim = self._module.rep_dim
        self.num_slots = self._module.num_slots

    def _init_optimizer(self):
        # optimizer
        if hasattr(self._config, "learning"):
            if hasattr(self._config.learning, "lr"):
                self._opt = torch.optim.Adam(
                    self._module.parameters(), lr=self._config.learning.lr
                )

    def __call__(self, obs: Tensor) -> Tensor:
        return self._module(obs)

    def forward(self, observations: Tensor) -> Tensor:
        return self(observations)

    def get_loss(self, obs: Tensor, with_rep=False) -> dict:
        return self._module.get_loss(obs, with_rep)

    def train(self, training=True) -> None:
        if training:
            self._module.train()
        else:
            self.eval()
        return None

    def eval(self) -> None:
        self._module.eval()
        return None

    def to(self, device: str) -> None:
        self._module.to(device)
        if hasattr(self, "_opt"):
            optimizer_to(self._opt, device)

    def set_zero_grad(self):
        if hasattr(self, "_opt"):
            self._opt.zero_grad()

    def do_step(self):
        if hasattr(self, "_opt"):
            self._opt.step()

    def get_samples(self, obs: Tensor, prev_slots=None) -> dict:
        return self._module.get_samples(obs, prev_slots=prev_slots)

    def update(self, obs: Tensor, masks, step: int) -> dict:
        if hasattr(self, "_opt"):
            self._opt.zero_grad()
            metrics = self.get_loss(obs, masks)
            metrics["loss"].backward()
            if hasattr(self._config.learning, "clip"):
                clip_norm_type = self._config.learning.clip_norm_type if hasattr(self._config.learning, "clip_norm_type") else "inf"
                norm = clip_grad_norm_(
                    self._module.parameters(), self._config.learning.clip, clip_norm_type
                )
                metrics["norm"] = norm
            self._opt.step()
            return metrics
        else:
            return {}

    def save(self) -> dict:
        checkpoint = {}
        checkpoint["ocr_module_state_dict"] = self._module.state_dict()
        if hasattr(self, "_opt"):
            checkpoint["ocr_opt_state_dict"] = self._opt.state_dict()
        return checkpoint

    def load(self, checkpoint: str) -> None:
        self._module.load_state_dict(checkpoint["ocr_module_state_dict"])
        # for name1, name2 in zip(self._module.state_dict().keys(), checkpoint["ocr_nets_state_dict"].keys()):
        #    self._module.state_dict()[name1] = checkpoint["ocr_nets_state_dict"][name2]
        if hasattr(self, "_opt"):
            self._opt.load_state_dict(checkpoint["ocr_opt_state_dict"])
