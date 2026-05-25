# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
import numpy as np

from typing import Any

class RewardShapingWrapper(gym.Wrapper):
    """
    Reward Shaping de Grado Profesional para circuitos aleatorios.
    Filosofía: Premiar la VELOCIDAD + PROGRESO SOSTENIDO. Castigar el ESTANCAMIENTO.
    NO castigamos el volante directamente (causa Reward Hacking hacia la hierba).
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.frames_without_progress = 0
        self.consecutive_progress_frames = 0
        self.total_progress_frames = 0  # Total de frames en pista en este episodio

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)

        steering = float(action[0])
        gas     = float(action[1])
        brake   = float(action[2])

        # ================================================================
        # BLOQUE 1: PROGRESO EN PISTA (reward > 0 = pisando asfalto nuevo)
        # ================================================================
        if reward > 0:
            self.frames_without_progress = 0
            self.consecutive_progress_frames += 1
            self.total_progress_frames += 1

            reward += gas * 0.15  # Premio por velocidad sostenida

            in_curve    = abs(steering) > 0.1
            smooth_turn = abs(steering) < 0.6
            light_brake = brake < 0.3
            if in_curve and smooth_turn and light_brake:
                reward += 0.4  # Trazada limpia: el Apex

            # Premio por racha larga sin salirse (supervivencia sostenida)
            if self.consecutive_progress_frames > 200:
                reward += 0.2

        else:
            self.consecutive_progress_frames = 0
            self.frames_without_progress += 1

            penalty = min(self.frames_without_progress * 0.003, 1.0)
            reward -= penalty

            if self.frames_without_progress >= 300:
                terminated = True
                reward -= 15

        # --- PREMIO POR COMPLETAR EL CIRCUITO ---
        # CarRacing termina naturalmente (terminated=True) cuando el coche completa el 95% del circuito.
        # Si la muerte NO fue por nuestra regla (frames<300) sino por terminación natural,
        # significa que completó la pista. Es el logro más grande que podemos premiar.
        if terminated and self.frames_without_progress < 300:
            laps_bonus = min(self.total_progress_frames * 0.05, 100.0)
            reward += laps_bonus  # Premio proporcional al recorrido completado

        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        self.frames_without_progress = 0
        self.consecutive_progress_frames = 0
        self.total_progress_frames = 0
        return self.env.reset(**kwargs)


def main() -> None:
    base_env = gym.make("CarRacing-v3", render_mode="human")
    env_state = gym.wrappers.GrayscaleObservation(base_env, keep_dim=False)
    env_stack = gym.wrappers.FrameStackObservation(env_state, stack_size=4)
    env = RewardShapingWrapper(env_stack)

    observation, info = env.reset(seed=42)
    print("Environment initialized!")
    print("Observation shape:", observation.shape)

    for _ in range(1000):
        random_action: Any = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random_action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
