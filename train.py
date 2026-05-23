import os

# Apagamos los logs ruidosos de TensorFlow ANTES de importar cualquier librería pesada
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from stable_baselines3 import PPO
# pyrefly: ignore [missing-import]
from stable_baselines3.common.vec_env import SubprocVecEnv
# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import os
import glob

class KeepLatestCheckpointsCallback(BaseCallback):
    """
    Callback personalizado para borrar modelos viejos y evitar llenar el disco duro.
    Mantiene únicamente los últimos 'keep_n' archivos .zip
    """
    def __init__(self, save_path: str, keep_n: int = 4, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.keep_n = keep_n

    def _on_step(self) -> bool:
        # Revisamos la carpeta cada 10,000 pasos para limpiar la basura
        if self.n_calls % 10000 == 0:
            list_of_files = glob.glob(os.path.join(self.save_path, "*.zip"))
            if len(list_of_files) > self.keep_n:
                # Ordenar por fecha de creación/modificación (los más viejos primero)
                list_of_files.sort(key=os.path.getctime)
                # Borrar todos excepto los últimos 'keep_n'
                for file_to_delete in list_of_files[:-self.keep_n]:
                    try:
                        os.remove(file_to_delete)
                    except OSError:
                        pass
        return True

# Importamos las piezas arquitectónicas (Wrappers) que construimos en la Fase 2 y 3
from env_setup import RewardShapingWrapper

# pyrefly: ignore [missing-import]
from stable_baselines3.common.monitor import Monitor

def make_env(rank: int, seed: int = 42):
    """
    Función generadora para crear clones de nuestro entorno.
    """
    def _init() -> gym.Env:
        # Volvemos al entrenamiento fantasma (sin pantalla) para evitar el "juego pillado" en Windows.
        # Usa enjoy.py en otra consola si quieres auditar la conducción.
        base_env = gym.make("CarRacing-v3", render_mode=None)
        # Le damos los "ojos" reales a la IA usando GrayScale
        env_state = gym.wrappers.GrayscaleObservation(base_env, keep_dim=True)
        env = RewardShapingWrapper(env_state)
        
        # EL SECRETO PARA TENSORBOARD: El wrapper Monitor
        # SB3 necesita este wrapper para llevar la cuenta de la puntuación total 
        # del episodio y poder graficar "ep_rew_mean" (Recompensa Media).
        env = Monitor(env)
        
        env.reset(seed=seed + rank) 
        return env
    return _init

import os

def train() -> None:
    # --- ARQUITECTURA MULTI-AGENTE (Tu idea implementada) ---
    num_cars = 6  # Levantaremos 6 universos paralelos (ajusta según los núcleos de tu CPU)
    print(f"Desplegando {num_cars} coches en paralelo usando Multiprocesamiento...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cars)])
    
    # --- CONFIGURACIÓN DE APRENDIZAJE CONTINUO ---
    CONTINUAR_ENTRENAMIENTO = True
    MODELO_BASE = "ppo_carracing_final"  # Sin la extensión .zip
    
    if CONTINUAR_ENTRENAMIENTO and os.path.exists(MODELO_BASE + ".zip"):
        # El "Por qué": En lugar de inicializar la red neuronal con pesos aleatorios,
        # cargamos los pesos (conocimientos) del modelo anterior.
        # Es vital pasarle el 'env=vec_env' para que sepa en qué pista va a seguir corriendo.
        print(f"🧠 Cargando cerebro base desde '{MODELO_BASE}'...")
        model = PPO.load(MODELO_BASE, env=vec_env)
        # Nota: Al cargar, PPO recuerda su learning_rate y estado previo.
    else:
        print("🌱 Iniciando un cerebro completamente nuevo (Tabula Rasa)...")
        # --- HIPERPARÁMETROS PPO (El cerebro de la IA) ---
        model = PPO(
            policy="CnnPolicy", # Cambiamos la red a Convolucional (ojos)
            env=vec_env,
            learning_rate=3e-4,
            batch_size=256,
            verbose=1,
            tensorboard_log="./ppo_carracing_tensorboard/"
        )
    
    # Callback: Guarda el modelo cada 10,000 pasos en caso de que se apague la PC
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000, 
        save_path="./models/", 
        name_prefix="ppo_car"
    )
    
    # Nuevo Callback: El limpiador automático (Solo guarda los últimos 4)
    cleanup_callback = KeepLatestCheckpointsCallback(save_path="./models/", keep_n=4)
    
    # Empaquetamos los callbacks juntos
    callbacks = CallbackList([checkpoint_callback, cleanup_callback])
    
    print("¡Iniciando Entrenamiento INFINITO! (Pulsa Ctrl+C cuando quieras detenerlo y guardar)")
    
    try:
        # Bucle infinito: entrena en bloques de 100k pasos hasta que tú lo detengas
        while True:
            model.learn(
                total_timesteps=100_000, 
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name="Coche_Continuo"
            )
            # Guardado automático al final de cada bloque por seguridad
            model.save(MODELO_BASE)
            
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento detenido manualmente por el Arquitecto.")
    
    print(f"Guardando el cerebro final en '{MODELO_BASE}.zip'...")
    model.save(MODELO_BASE)
    
    try:
        vec_env.close()
    except Exception:
        # En Windows, pulsar Ctrl+C a veces mata los subprocesos antes de que el padre pueda
        # cerrarlos ordenadamente, lanzando un BrokenPipeError. Lo ignoramos por ser inofensivo.
        pass

if __name__ == "__main__":
    train()
