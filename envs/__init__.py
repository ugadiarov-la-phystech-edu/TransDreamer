import time

import cloudpickle
import numpy as np
import portal
import torch

from ocr.slate.slate_extractor import SLATEExtractor
from .atari_env import OneHotAction, TimeLimit, Collect, RewardObs
from .atari_env import Atari
from .crafter import Crafter
from .tools import count_episodes, save_episodes, video_summary
import pathlib
import pdb
import json

def count_steps(datadir, cfg):
  return tools.count_episodes(datadir)[1]

def summarize_episode(episode, config, datadir, writer, prefix):
  episodes, steps = tools.count_episodes(datadir)
  length = (len(episode['reward']) - 1) * config.env.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'{prefix}/episodes', episodes)]
  step = count_steps(datadir, config)
  env_step = step * config.env.action_repeat
  with (pathlib.Path(config.logdir) / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', env_step)] + metrics)) + '\n')
  [writer.add_scalar('sim/' + k, v, env_step) for k, v in metrics]
  tools.video_summary(writer, f'sim/{prefix}/video', episode['image'][None, :1000], env_step)

  if 'episode_done' in episode:
    episode_done = episode['episode_done']
    num_episodes = sum(episode_done)
    writer.add_scalar(f'sim/{prefix}/num_episodes', num_episodes, env_step)
    # compute sub-episode len
    episode_done = np.insert(episode_done, 0, 0)
    episode_len_ = np.where(episode_done)[0]
    if len(episode_len_) > 0:
      if len(episode_len_) > 1:
        episode_len_ = np.insert(episode_len_, 0, 0)
        episode_len_ = episode_len_[1:] - episode_len_[:-1]
        writer.add_histogram(f'sim/{prefix}/sub_episode_len', episode_len_, env_step)
        writer.add_scalar(f'sim/{prefix}/sub_episode_len_min', episode_len_[1:].min(), env_step)
        writer.add_scalar(f'sim/{prefix}/sub_episode_len_max', episode_len_[1:].max(), env_step)
        writer.add_scalar(f'sim/{prefix}/sub_episode_len_mean', episode_len_[1:].mean(), env_step)
        writer.add_scalar(f'sim/{prefix}/sub_episode_len_std', episode_len_[1:].std(), env_step)

  writer.flush()


def summarize(global_step, n_episodes, episode_stats, config, writer, prefix):
  env_step = global_step * config.env.action_repeat
  for episode_stat in episode_stats:
    length = episode_stat['length'] * config.env.action_repeat
    ret = episode_stat['return']
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
      (f'{prefix}/return', float(ret)),
      (f'{prefix}/length', float(length)),
      (f'{prefix}/episodes', n_episodes)]
    with (pathlib.Path(config.logdir) / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps(dict([('step', env_step)] + metrics)) + '\n')

  agg_metrics = [
      (f'{prefix}/return', np.mean([e['return'] for e in episode_stats])),
      (f'{prefix}/length', np.mean([e['length'] for e in episode_stats])),
      (f'{prefix}/episodes', n_episodes)]
  [writer.add_scalar('sim/' + k, v, env_step) for k, v in agg_metrics]
  writer.flush()


def make_env(cfg, datadir, store, seed=0):

  suite, task = cfg.env.name.split('_', 1)

  if suite == 'atari':
    env = Atari(
        task, cfg.env.action_repeat, (64, 64), grayscale=cfg.env.grayscale,
        life_done=False, sticky_actions=True, seed=seed, all_actions=cfg.env.all_actions)
    env = OneHotAction(env)

  elif suite == 'crafter':
    env = Crafter(task, (64, 64), seed)
    env = OneHotAction(env)

  elif suite == 'shapes2d':
    import envs.shapes2d
    from envs.from_gym import FromGym
    env = FromGym(task, cfg.env.size, seed=seed)
    env = OneHotAction(env)

  else:
    raise NotImplementedError(suite)

  env = TimeLimit(env, cfg.env.time_limit, cfg.env.time_penalty)

  env = Collect(env, callbacks=None, precision=cfg.env.precision)
  env = RewardObs(env)

  return env


class BatchEnv:
  def __init__(self, make_env_fns, parallel, input_type, device='cpu'):
    self.parallel = parallel
    self.n_envs = len(make_env_fns)
    self.device = device
    self.input_type = input_type
    if self.parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(len(make_env_fns))])
      self.stop = context.Event()
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('action_space',))
      self.action_space = self._receive(self.pipes[0])
      self.pipes[0].send(('observation_space',))
      self.observation_space = self._receive(self.pipes[0])
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.action_space = self.envs[0].action_space
      self.observation_space = self.envs[0].observation_space

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(stop, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while not stop.is_set():
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          step_result = env.step(act)
          pipe.send(('result', step_result))
        elif msg == 'reset':
          assert len(args) == 0
          obs = env.reset()
          pipe.send(('result', obs))
        elif msg == 'render':
          assert len(args) == 1
          mode = args[0]
          image = env.render(mode)
          pipe.send(('result', image))
        elif msg == 'observation_space':
          assert len(args) == 0
          pipe.send(('result', env.observation_space))
        elif msg == 'action_space':
          assert len(args) == 0
          pipe.send(('result', env.action_space))
        elif msg == 'close':
          assert len(args) == 0
          break
        else:
          raise ValueError(f'Invalid message {msg}')
    except ConnectionResetError:
      print('Connection to driver lost')
    except Exception as e:
      pipe.send(('error', e))
      raise
    finally:
      try:
        env.close()
      except Exception:
        pass
      pipe.close()

  def reset(self, env_ids=None):
    if env_ids is None:
        env_ids = range(self.n_envs)

    if self.parallel:
      [self.pipes[env_id].send(('reset',)) for env_id in env_ids]
      obs = [self._receive(self.pipes[env_id]) for env_id in env_ids]
    else:
      obs = [self.envs[env_id].reset() for env_id in env_ids]

    obs = {k: [o[k] for o in obs] for k in obs[0]}
    obs[self.input_type] = torch.as_tensor(np.concatenate(obs[self.input_type], axis=0), device=self.device)
    return obs

  def step(self, acts):
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      step_results = [self._receive(pipe) for pipe in self.pipes]
    else:
      step_results = [env.step(act) for env, act in zip(self.envs, acts)]

    obs, reward, done, info = zip(*step_results)
    obs = {k: [o[k] for o in obs] for k in obs[0]}
    obs[self.input_type] = torch.as_tensor(np.concatenate(obs[self.input_type], axis=0), device=self.device)

    return obs, reward, done, info

  def close(self):
    if self.parallel:
      [proc.kill() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def sample_random_action(self, n_envs=None):
    if n_envs is None:
        n_envs = self.n_envs

    action = np.zeros((n_envs, self.action_space.n,), dtype=np.float)
    idx = np.random.randint(0, self.action_space.n, size=(n_envs,))
    action[np.arange(n_envs), idx] = 1
    return action


class SlotBatchEnv(BatchEnv):
  def __init__(self, make_env_fns, parallel, input_type, on_episode_end, cfg, device='cpu'):
    super(SlotBatchEnv, self).__init__(make_env_fns, parallel, 'image', device)
    assert input_type == 'slot', f'{input_type} != slot'
    self.device = device
    self.slot_input_type = input_type
    self.slot_extractor = SLATEExtractor(cfg.config_path, cfg.checkpoint_path, cfg.image_size, device)
    self.on_episode_end = on_episode_end
    self._episode_slots = [None] * self.n_envs

    import gym
    self.observation_space = gym.spaces.Dict({
        'slot': gym.spaces.Box(low=0, high=255, shape=(self.slot_extractor.n_slots, self.slot_extractor.dim), dtype=np.float32),
        'reward': self.observation_space['reward'],
    })

  def _generate_slots(self, env_ids=None):
    env_ids = range(self.n_envs) if env_ids is None else env_ids
    obss = [self.observation_space.sample() for _ in env_ids]
    obss = {k: [o[k] for o in obss] for k in obss[0]}
    slot = torch.as_tensor(np.stack(obss[self.slot_input_type], axis=0), device=self.device)

    return slot

  def reset(self, env_ids=None):
    if env_ids is None:
        env_ids = range(self.n_envs)

    obs = super(SlotBatchEnv, self).reset(env_ids)
    images = obs.pop('image')
    slots = self.slot_extractor.get_slots(images, previous_slots=None)
    for i, env_id in enumerate(env_ids):
      self._episode_slots[env_id] = [slots[i]]

    obs[self.slot_input_type] = slots

    return obs

  def step(self, acts):
    obss, rewards, dones, infos = super(SlotBatchEnv, self).step(acts)
    images = obss.pop('image')
    slots = self.slot_extractor.get_slots(images, previous_slots=torch.stack([slot_history[-1] for slot_history in self._episode_slots]))
    for i in range(self.n_envs):
      self._episode_slots[i].append(slots[i])

    for i, done in enumerate(dones):
      if done:
        episode = infos[i]['episode']
        infos[i]['episode'] = episode.pop('stats')
        del episode['image']
        episode['slot'] = torch.stack(self._episode_slots[i]).detach().cpu().numpy()
        self.on_episode_end(episode, i)

    obss[self.slot_input_type] = slots

    return obss, rewards, dones, infos
