import os

# Apagamos los logs ruidosos de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from stable_baselines3 import PPO
from env_setup import RewardShapingWrapper
import time

def enjoy() -> None:
    # 1. Cargamos el entorno original.
    # ESTA VEZ SÍ usamos render_mode="human" para poder ver a nuestra IA conduciendo.
    base_env = gym.make("CarRacing-v3", render_mode="human")
    env_state = gym.wrappers.GrayscaleObservation(base_env, keep_dim=True)
    env = RewardShapingWrapper(env_state)
    
    print("Cargando el cerebro del coche...")
    try:
        # 2. Cargamos el modelo entrenado desde el archivo generado por train.py
        model = PPO.load("ppo_carracing_final")
    except FileNotFoundError:
        print("¡Error! No se encontró el cerebro. Asegúrate de ejecutar train.py primero.")
        return
        
    obs, info = env.reset()
    print("¡Piloto automático activado! Disfruta del show.")
    
    # 3. Bucle de Inferencia (Inference Loop)
    while True:
        # A diferencia del sample() aleatorio, aquí la Red Neuronal toma la decisión.
        # deterministic=True significa que el modelo tomará la acción que considere más óptima,
        # sin aplicar ruido de exploración aleatorio.
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Una pequeña pausa para que los humanos podamos ver los fotogramas a buena velocidad
        time.sleep(0.02)
        
        if terminated or truncated:
            print("Pista terminada o coche estancado. Reseteando...")
            obs, info = env.reset()

if __name__ == "__main__":
    enjoy()
