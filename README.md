# FacialRecognition-Students
Este repositorio se enfoca en el desarrollo de un sistema basado en redes neuronales para detectar emociones en estudiantes universitarios en el Perú. El objetivo principal es mejorar la calidad de la educación y el rendimiento de los estudiantes al proporcionar información inteligente sobre sus emociones utilizando redes neuronales.

## Instrucciones de instalación y uso

1. Descarga el repositorio como un archivo ZIP desde la siguiente dirección: [https://github.com/Elvis-Benites-N/FacialRecognition-Students.git](https://github.com/Elvis-Benites-N/FacialRecognition-Students.git) o clonal el repositorio con el siguiente comando:
```bash
git clone https://github.com/Elvis-Benites-N/FacialRecognition-Students.git
```
3. Descomprime el archivo descargado. Luego, descomprime el archivo `data.zip`, lo cual debería generar una carpeta con la siguiente estructura:

```
.
├── data
│   ├── haarcascade_frontalface_default.xml
│   ├── ...
├── main.py
├── model_file.h5
├── requirements.txt
├── test.py
├── ...
```

3. Asegúrate de tener instalado Python 3.8.10 en tu sistema. Puedes descargar la versión correspondiente desde la página oficial de Python: [https://www.python.org/downloads/release/python-3810/](https://www.python.org/downloads/release/python-3810/)

4. Entra en la ubicación raíz del proyecto descargado y ejecuta el siguiente comando para verificar que la versión de Python sea la correcta:

```bash
python --version
```

La versión debe ser 3.10.0.

5. Instala las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```

Con esta versión de Python no debería haber conflictos.

## Uso

- El archivo principal `main.py` se encarga de entrenar el modelo y generará un archivo llamado `model_file.h5`, el cual contendrá el modelo entrenado.

- Para realizar la detección de emociones en tiempo real a través de la cámara, utiliza el archivo `test.py`. Este archivo utiliza la librería `haarcascade_frontalface_default.xml` para detectar los rostros en las imágenes capturadas. Luego, envía estos rostros al modelo entrenado y realiza la detección de emociones. 

  Para ejecutar el archivo de detección, ingresa el siguiente comando:

  ```bash
  python test.py
  ```

  Para cerrar la ventana de detección, presiona la tecla "q".

Recuerda que el archivo `model_file.h5` contiene el modelo entrenado y puedes utilizarlo para realizar la detección de emociones en otras imágenes o aplicaciones.

¡Disfruta del proyecto!

