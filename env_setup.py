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
        self.consecutive_progress_frames = 0  # Racha de éxito continuo

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

            # Premio por velocidad: cuanto más gas mientras avanza, mejor.
            # Incentiva no ir demasiado despacio en las rectas.
            speed_bonus = gas * 0.15
            reward += speed_bonus

            # Premio por trazada suave en curva:
            # Condición: girando (> 0.1), sin derrapar (< 0.6), freno suave (< 0.3).
            # Usamos < 0.3 en lugar de == 0.0 para que funcione con acciones continuas reales.
            in_curve    = abs(steering) > 0.1
            smooth_turn = abs(steering) < 0.6
            light_brake = brake < 0.3
            if in_curve and smooth_turn and light_brake:
                reward += 0.4  # Trazada limpia: el Apex

            # Premio por racha: si lleva mucho tiempo avanzando sin salirse,
            # le damos un pequeño bonus acumulativo que fomenta completar circuitos enteros.
            if self.consecutive_progress_frames > 100:
                reward += 0.1

        # ================================================================
        # BLOQUE 2: ESTANCAMIENTO (reward <= 0 = hierba, frenado, girando sin avanzar)
        # ================================================================
        else:
            self.consecutive_progress_frames = 0
            self.frames_without_progress += 1

            # Castigo progresivo muy suave al principio (no asustamos al coche en curvas lentas)
            # pero que crece hasta un máximo de 1.0 para evitar que se quede parado indefinidamente.
            penalty = min(self.frames_without_progress * 0.003, 1.0)
            reward -= penalty

            # MUERTE SÚBITA: Aumentamos a 300 frames (~6 segundos a 50 FPS).
            # Crucial para circuitos aleatorios con curvas muy cerradas donde el coche
            # necesita frenar mucho y el progreso es naturalmente lento.
            if self.frames_without_progress >= 300:
                terminated = True
                reward -= 15  # Penalización de salida clara, pero no catastrófica

        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        self.frames_without_progress = 0
        self.consecutive_progress_frames = 0
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
