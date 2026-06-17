import gymnasium as gym  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
from typing import Any


class EarlyStopWrapper(gym.Wrapper):
    """
    Wrapper minimalista: SOLO termina el episodio si el coche se estanca.
    NO modificamos la recompensa. Dejamos que VecNormalize haga ese trabajo.
    La senal nativa de CarRacing (+1000/N por tile, -0.1 por frame) es suficiente.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.neg_reward_counter = 0

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward < 0:
            self.neg_reward_counter += 1
            # Con frame_skip=2, 150 frames = 300 frames del motor = ~6 segundos.
            if self.neg_reward_counter > 150:
                terminated = True
        else:
            self.neg_reward_counter = 0

        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        self.neg_reward_counter = 0
        return self.env.reset(**kwargs)


def build_env(render_mode: str = None) -> gym.Env:
    """
    Cadena de wrappers alineada con RL Baselines3 Zoo.
    Orden: Base -> FrameSkip -> Resize -> Grayscale -> EarlyStop
    """
    base_env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = gym.wrappers.MaxAndSkipObservation(base_env, skip=2)
    env = gym.wrappers.ResizeObservation(env, shape=(64, 64))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    env = EarlyStopWrapper(env)
    return env
