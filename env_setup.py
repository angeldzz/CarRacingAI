# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
import numpy as np

from typing import Any
    
# Hemos eliminado el antiguo StateExtractionWrapper que cegaba a la IA.
# Ahora usamos gym.wrappers.GrayscaleObservation directamente en la cadena de Wrappers.

class RewardShapingWrapper(gym.Wrapper):
    """
    Fase 3: Reward Shaping Avanzado.
    Interceptamos la acción y la recompensa original para crear castigos inteligentes.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Llevamos la cuenta de cuánto tiempo lleva atascado o en el césped
        self.frames_without_progress = 0

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # En CarRacing, action es un array continuo: [volante, gas, freno]
        gas = float(action[1])
        brake = float(action[2])
        
        # CarRacing da puntos positivos solo si pisas asfalto nuevo.
        if reward <= 0:
            # NO está pisando pista nueva (está en césped, yendo al revés, o parado)
            self.frames_without_progress += 1
            
            # Castigo natural base
            reward -= 0.1 
            
            # LA LAVA (Castigo Progresivo Limitado):
            # Calculamos la penalización, pero la "topamos" (cap) para que no sea infinita.
            # Si el castigo es demasiado grande matemáticamente (-3000 pts), la Red Neuronal colapsa.
            penalty = self.frames_without_progress * 0.05
            if penalty > 2.0:
                penalty = 2.0
            reward -= penalty
            
            # MUERTE SÚBITA (Early Stopping):
            # Si lleva 50 fotogramas (aprox 1 segundo) fuera de la pista sin hacer progreso,
            # cortamos el episodio. Es mejor reiniciar que dejarlo acumular miles de puntos negativos.
            if self.frames_without_progress >= 50:
                terminated = True
                reward -= 20  # Castigo final por perderse

            
        else:
            # ¡Está pisando asfalto nuevo (progresando)! 
            # Reseteamos el contador porque ha vuelto a la zona segura
            self.frames_without_progress = 0
            
            # Solo aquí, en la zona segura, le damos puntos extra por acelerar
            reward += (gas * 0.1)
            
        return obs, float(reward), terminated, truncated, info

def main() -> None:
    # Por qué CarRacing-v3: Es la versión más actual del entorno continuo de Gymnasium para coches.
    # Usamos render_mode="human" para poder auditar visualmente el comportamiento del agente.
    base_env = gym.make("CarRacing-v3", render_mode="human")
    
    # Aplicamos nuestros Wrappers al entorno base en formato de "Cebolla" (Capas)
    # 1. Extraemos el estado visual (Fase 2 - Redux)
    # keep_dim=True mantiene la forma (96, 96, 1) en lugar de (96, 96), lo cual es crítico para CnnPolicy
    env_state = gym.wrappers.GrayscaleObservation(base_env, keep_dim=True)
    # 2. Modificamos las recompensas (Fase 3)
    env = RewardShapingWrapper(env_state)
    
    # Por qué resetear: Los Procesos de Decisión de Markov (MDP) requieren un estado inicial.
    # El seed=42 fija la aleatoriedad de la pista para garantizar reproducibilidad en pruebas.
    observation, info = env.reset(seed=42)

    print("Environment initialized! Starting random action loop...")
    print("Action space:", env.action_space)
    print("Observation space (Nuevo Estado 1D):", env.observation_space)
    print("Forma real de la observación actual:", observation.shape)
    
    # Por qué acciones aleatorias: Es la "baseline" de exploración. Si el agente no puede 
    # descubrir estados valiosos con ruido, el algoritmo de RL fallará.
    for _ in range(1000):
        # Sample toma un vector aleatorio de 3 elementos continuos: [volante, gas, freno].
        random_action: Any = env.action_space.sample()
        
        # El motor avanza 1 frame. Nos devuelve:
        # observation: Ahora es nuestro vector 1D de telemetría.
        # reward: Función de recompensa (la señal que el modelo intentará maximizar)
        # terminated/truncated: Condiciones de fin (salida de pista, victoria o timeout)
        observation, reward, terminated, truncated, info = env.step(random_action)
        
        if terminated or truncated:
            print("Episode finished. Resetting...")
            observation, info = env.reset()

    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
