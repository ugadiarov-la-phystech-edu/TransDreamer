import threading

import gym
import numpy as np
from gym.wrappers import ResizeObservation


class FromGym:

  LOCK = threading.Lock()

  def __init__(
      self, name, size=(64, 64), seed=0):
    if isinstance(size, int):
      size = (size, size)

    assert size[0] == size[1]
    with self.LOCK:
      env = gym.make(name, seed=seed)

    env = ResizeObservation(env, size)
    self._env = env
    self._size = size

  @property
  def observation_space(self):
    shape = (3,) + self._size
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    image = np.transpose(image, (2, 0, 1)) # 3, 64, 64
    return {'image': image}

  def step(self, action):
    image, reward, done, info = self._env.step(action)
    image = np.transpose(image, (2, 0, 1)) # 3, 64, 64
    obs = {'image': image}
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)
