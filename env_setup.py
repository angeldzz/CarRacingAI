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
        steering = float(action[0])
        gas = float(action[1])
        brake = float(action[2])
        
        # --- NUEVO: REWARD HACKING SOLUCIONADO ---
        # Lección RL: Si castigamos el volante siempre, el coche preferirá irse 
        # recto a la hierba para evitar el castigo. ¡Solo castigamos el DERRAPE!
        steering_penalty = 0.0
        if gas > 0.5 and abs(steering) > 0.5:
            steering_penalty = 0.1  # Solo hay multa si intentas tomar curvas cerradas acelerando a fondo
            
        reward -= steering_penalty

        # CarRacing da puntos positivos solo si pisas asfalto nuevo.
        if reward <= 0:
            self.frames_without_progress += 1
            
            # LA LAVA (Castigo Progresivo Limitado):
            penalty = self.frames_without_progress * 0.005
            if penalty > 1.0:
                penalty = 1.0
            reward -= penalty
            
            # MUERTE SÚBITA (Early Stopping):
            if self.frames_without_progress >= 200:
                terminated = True
                reward -= 20  # Castigo por perderse definitivamente

        else:
            # ¡Está pisando asfalto nuevo (progresando)! 
            self.frames_without_progress = 0
            
            # Premiamos ligeramente si va acelerando cuando lo hace bien (rectas)
            reward += (gas * 0.1)
            
            # --- NUEVO: PREMIO POR CURVA PERFECTA (EL APEX) ---
            # Si pisa asfalto nuevo, está girando el volante (está en una curva),
            # NO está frenando (mantiene la inercia) y NO está derrapando (volante suave <= 0.5)
            if 0.15 < abs(steering) <= 0.5 and brake == 0.0:
                reward += 0.5  # Premio por trazar de forma suave y eficiente
            
        return obs, float(reward), terminated, truncated, info

def main() -> None:
    # Por qué CarRacing-v3: Es la versión más actual del entorno continuo de Gymnasium para coches.
    # Usamos render_mode="human" para poder auditar visualmente el comportamiento del agente.
    base_env = gym.make("CarRacing-v3", render_mode="human")
    
    # 1. Extraemos el estado visual (Fase 2 - Redux)
    # keep_dim=False quita el canal extra, dejando la imagen 2D (96, 96)
    env_state = gym.wrappers.GrayscaleObservation(base_env, keep_dim=False)
    
    # 2. Frame Stacking: Apilamos los últimos 4 fotogramas para dotar al coche
    # de inercia y velocidad (Violación de Markov resuelta). Forma final: (4, 96, 96)
    env_stack = gym.wrappers.FrameStackObservation(env_state, stack_size=4)
    
    # 3. Modificamos las recompensas (Fase 3)
    env = RewardShapingWrapper(env_stack)
    
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
