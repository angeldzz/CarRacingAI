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
        # Le damos los "ojos" reales a la IA usando GrayScale (sin dimensión extra)
        env_state = gym.wrappers.GrayscaleObservation(base_env, keep_dim=False)
        
        # Frame Stacking: Apilamos 4 fotogramas para que el coche tenga memoria temporal
        env_stack = gym.wrappers.FrameStackObservation(env_state, stack_size=4)
        
        env = RewardShapingWrapper(env_stack)
        
        # EL SECRETO PARA TENSORBOARD: El wrapper Monitor
        # SB3 necesita este wrapper para llevar la cuenta de la puntuación total 
        # del episodio y poder graficar "ep_rew_mean" (Recompensa Media).
        env = Monitor(env)
        
        env.reset(seed=seed + rank) 
        return env
    return _init

import os

def train() -> None:
    # --- ARQUITECTURA MULTI-AGENTE ---
    # Ryzen 5 3600 tiene 12 hilos lógicos. 12 es el límite físico perfecto.
    num_cars = 12  # Óptimo para Ryzen 5 3600 (6 cores / 12 threads)
    print(f"Desplegando {num_cars} coches en paralelo usando Multiprocesamiento...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cars)])
    
    # --- CONFIGURACIÓN DE APRENDIZAJE CONTINUO ---
    # ¡ATENCIÓN ARQUITECTO! Como hemos cambiado los ojos a 4 canales (FrameStack),
    # el cerebro anterior ya no nos sirve. Debemos iniciar desde cero.
    CONTINUAR_ENTRENAMIENTO = True     
    MODELO_BASE = "ppo_carracing_final"  # Sin la extensión .zip
    
    if CONTINUAR_ENTRENAMIENTO and os.path.exists(MODELO_BASE + ".zip"):
        # El "Por qué": En lugar de inicializar la red neuronal con pesos aleatorios,
        # cargamos los pesos (conocimientos) del modelo anterior.
        # Es vital pasarle el 'env=vec_env' para que sepa en qué pista va a seguir corriendo.
        print(f"🧠 Cargando cerebro base desde '{MODELO_BASE}' [MODO FINE-TUNING]...")
        # FASE DE FINE-TUNING (>4M pasos):
        # Reducimos ent_coef 0.01 -> 0.003: ya no necesitamos explorar, queremos explotar.
        # Un std=1.2 a 5M pasos indica que ent_coef alto impide converger hacia acciones precisas.
        # Reducimos learning_rate 3e-4 -> 1e-4: ajustes más finos, sin borrar lo ya aprendido.
        custom_args = {
            "n_steps": 4096,
            "batch_size": 4096,
            "ent_coef": 0.0,   # std=1.22 CONGELADO: dejamos de pagar por el caos
            "learning_rate": 1e-4,
        }
        model = PPO.load(MODELO_BASE, env=vec_env, custom_objects=custom_args)
    else:
        print("🌱 Iniciando un cerebro completamente nuevo (Tabula Rasa)...")
        # --- HIPERPARÁMETROS PPO (Modo GPU ) ---
        model = PPO(
            policy="CnnPolicy",
            env=vec_env,
            # Forzamos a que PyTorch intente usar la GPU. Si no tienes CUDA, esto dará error.
            device="cuda",
            learning_rate=3e-4,
            
            # n_steps: Cuántas transiciones recopila CADA clon antes de actualizar la red.
            # Al subirlo a 2048 (default) o 4096, le damos muchísimos más "datos posibles" 
            # de una sola vez a la GPU para que aprenda trayectorias largas de curvas.
            n_steps=4096,  # Sincronizado con custom_objects para consistencia total
            
            # batch_size: La cantidad de fotogramas que la GPU mastica de golpe.
            # En CPU 256 está bien, pero en GPU podemos subir a 1024 o 2048 para 
            # paralelizar cálculos matriciales masivos y aprender rapidísimo.
            batch_size=4096,
            
            # ent_coef: 0.01 para modelos nuevos (exploración), 0.003 en fine-tuning.
            ent_coef=0.01,
            
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
