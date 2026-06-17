import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gymnasium as gym  # pyrefly: ignore [missing-import]
from stable_baselines3 import PPO  # pyrefly: ignore [missing-import]
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize  # pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList  # pyrefly: ignore [missing-import]
from stable_baselines3.common.monitor import Monitor  # pyrefly: ignore [missing-import]
import glob

from env_setup import build_env


class KeepLatestCheckpointsCallback(BaseCallback):
    def __init__(self, save_path: str, keep_n: int = 4, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.keep_n = keep_n

    def _on_step(self) -> bool:
        if self.n_calls % 10000 == 0:
            files = sorted(glob.glob(os.path.join(self.save_path, "*.zip")), key=os.path.getctime)
            for f in files[:-self.keep_n]:
                try:
                    os.remove(f)
                except OSError:
                    pass
        return True


def make_env(rank: int, seed: int = 42):
    def _init() -> gym.Env:
        env = build_env(render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train() -> None:
    NUM_ENVS = 8
    MODELO_BASE = "ppo_carracing_v3"
    CONTINUAR = os.path.exists(MODELO_BASE + ".zip")

    print(f"Desplegando {NUM_ENVS} coches en paralelo...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)

    if CONTINUAR:
        print(f"[LOAD] Cargando cerebro desde '{MODELO_BASE}'...")
        model = PPO.load(MODELO_BASE, env=vec_env)
        vec_norm_path = f"{MODELO_BASE}_vecnorm.pkl"
        if os.path.exists(vec_norm_path):
            vec_env = VecNormalize.load(vec_norm_path, vec_env)
    else:
        print("[NEW] Tabula Rasa con hiperparametros corregidos...")
        model = PPO(
            policy="CnnPolicy",
            env=vec_env,
            device="cuda",
            learning_rate=1e-4,     # 3e-4 causaba clip_fraction=0.73. Reducido a 1e-4.
            n_steps=512,
            batch_size=128,
            n_epochs=4,             # 10 epochs causaban KL=1.0. Reducido a 4.
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,          # 0.0 causaba std colapso (0.99->0.10). 0.01 lo previene.
            max_grad_norm=0.5,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./ppo_carracing_tensorboard/"
        )

    checkpoint_cb = CheckpointCallback(save_freq=10_000, save_path="./models/", name_prefix="ppo_car")
    cleanup_cb = KeepLatestCheckpointsCallback(save_path="./models/", keep_n=4)
    callbacks = CallbackList([checkpoint_cb, cleanup_cb])

    print("Entrenamiento INFINITO iniciado. (Ctrl+C para detener y guardar)")

    try:
        while True:
            model.learn(
                total_timesteps=100_000,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name="Coche_v3"
            )
            model.save(MODELO_BASE)
            vec_env.save(f"{MODELO_BASE}_vecnorm.pkl")
    except KeyboardInterrupt:
        print("\n[STOP] Entrenamiento detenido por el Arquitecto.")

    print(f"Guardando cerebro en '{MODELO_BASE}.zip'...")
    model.save(MODELO_BASE)
    vec_env.save(f"{MODELO_BASE}_vecnorm.pkl")

    try:
        vec_env.close()
    except Exception:
        pass

if __name__ == "__main__":
    train()
