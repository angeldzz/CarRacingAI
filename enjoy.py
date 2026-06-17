import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gymnasium as gym  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
from stable_baselines3 import PPO  # pyrefly: ignore [missing-import]
from env_setup import build_env
import time
from collections import deque


def enjoy() -> None:
    env = build_env(render_mode="human")

    print("Cargando el cerebro del coche...")
    modelo = "ppo_carracing_v3"
    try:
        model = PPO.load(modelo)
    except FileNotFoundError:
        print(f"No se encontro '{modelo}.zip'. Ejecuta train.py primero.")
        return

    obs, info = env.reset()
    frame_stack = deque([obs] * 2, maxlen=2)

    print("Piloto automatico activado!")

    while True:
        stacked_obs = np.concatenate(list(frame_stack), axis=-1)
        action, _states = model.predict(stacked_obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        frame_stack.append(obs)

        time.sleep(0.02)

        if terminated or truncated:
            print("Pista terminada o coche estancado. Reseteando...")
            obs, info = env.reset()
            frame_stack = deque([obs] * 2, maxlen=2)

if __name__ == "__main__":
    enjoy()
