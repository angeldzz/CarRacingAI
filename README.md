# 🏎️ Proyecto: Inteligencia Artificial en CarRacing

¡Bienvenido a la documentación de tu proyecto de Conducción Autónoma! Este documento explica de forma sencilla qué hemos construido, cómo funciona la "magia" detrás de la Inteligencia Artificial y para qué sirve cada archivo de nuestro proyecto.

---

## 🧠 1. El Concepto Base: Aprendizaje por Refuerzo
Nuestra IA no aprende como los programas tradicionales (donde un humano le dice "si ves una curva, gira a la derecha"). Nuestra IA aprende **por ensayo y error**, como un perro aprendiendo trucos. A esto se le llama **Reinforcement Learning (Aprendizaje por Refuerzo)**.
- Si el coche lo hace bien (avanza por la pista), le damos una **Recompensa (+)**.
- Si el coche lo hace mal (se sale al pasto o se queda quieto), le damos un **Castigo (-)**.

---

## 🛠️ 2. Las Herramientas que Usamos
- **Gymnasium (`CarRacing-v3`)**: Es nuestro simulador de físicas. Es el "mundo" donde vive el coche. Se encarga de calcular la gravedad, el derrape de las llantas y dibujar la pista.
- **Stable-Baselines3 (SB3)**: Es la librería matemática que le da el cerebro a nuestro coche. Usa un algoritmo de última generación llamado **PPO (Proximal Policy Optimization)**.
- **PyTorch / TensorFlow**: Son los motores matemáticos que SB3 usa por detrás para procesar la Red Neuronal profunda.

---

## 📂 3. Anatomía de nuestro Código

Hemos dividido el proyecto en tres archivos principales para mantener el código limpio y ordenado como un verdadero Arquitecto de Software:

### 👁️ `env_setup.py` (Los Ojos y Las Reglas)
Este archivo se encarga de modificar el mundo *antes* de presentárselo a la IA. Tiene dos trabajos críticos:
1. **Filtro Blanco y Negro (`GrayscaleObservation`)**: Para que el cerebro procese las imágenes súper rápido, le quitamos los colores al juego. La IA ve el mundo a través de una cámara en blanco y negro.
2. **El Sistema Anti-Cobardía (`RewardShapingWrapper`)**: Modificamos las reglas originales del juego para añadir castigos inteligentes. Si la IA descubre que quedarse quieta evita que choque, nosotros interceptamos sus pedales y la castigamos por no acelerar. ¡Le enseñamos a ser valiente!

### 🏋️ `train.py` (El Gimnasio de Entrenamiento)
Aquí es donde ocurre la fuerza bruta matemática.
- **Clonación de Universos (`SubprocVecEnv`)**: Para acelerar el aprendizaje, levantamos **6 simulaciones al mismo tiempo** usando los distintos núcleos de tu procesador. 6 coches aprenden a la vez y comparten sus conocimientos en un cerebro central.
- **Cerebro Cíclico (`PPO.load`)**: El script es lo suficientemente inteligente para buscar si ya existe un archivo `.zip` de un entrenamiento anterior. Si lo encuentra, lo carga y sigue aprendiendo desde ahí (esto se llama *Fine-Tuning*). Si no, empieza de cero ("Tabula Rasa").
- **Entrenamiento Fantasma**: Todo ocurre sin interfaz gráfica para no colapsar Windows, ejecutándose a cientos de fotogramas por segundo.

### 🍿 `enjoy.py` (La Sala de Exhibición)
El entrenamiento genera un archivo llamado `ppo_carracing_final.zip` (¡El cerebro del coche!). Este script simplemente carga ese cerebro, enciende la pantalla (`render_mode="human"`) y te permite disfrutar viendo cómo tu IA conduce usando lo que ha aprendido.

---

## 📈 4. ¿Cómo auditar el progreso?
Para evitar que entrenes "a ciegas", hemos integrado **TensorBoard**. Mientras `train.py` está corriendo, recopila todos los datos del aprendizaje.

Puedes ver gráficas hermosas en tu navegador abriendo una terminal y ejecutando:
```bash
tensorboard --logdir ppo_carracing_tensorboard
```

- **ep_rew_mean (Recompensa Media)**: Es la gráfica más importante. Si la línea sube, ¡tu coche se está volviendo más inteligente!
- **entropy_loss**: Mide la "curiosidad" del coche. Al principio explorará mucho dando volantazos locos, pero poco a poco se estabilizará.

---

> *Este proyecto es un entorno vivo. Puedes ajustar los castigos, darle más horas de entrenamiento, y ver cómo el coche domina cada vez mejor las curvas complejas.*
