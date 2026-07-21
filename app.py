"""
App de Streamlit: Deteccion de rostros en tiempo real con Haar Cascade (OpenCV).
Grupo: ___________ - Maestria en Inteligencia Artificial Aplicada, UDLA.

Para ejecutar localmente:
    streamlit run app.py

Para desplegar en linea:
    Subir este archivo junto con requirements.txt a un repositorio de GitHub
    y desplegarlo en https://share.streamlit.io (Streamlit Community Cloud).
"""

import time
import numpy as np
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Deteccion de rostros - Haar Cascade", layout="centered")

st.title("Deteccion de rostros en tiempo real (Haar Cascade)")
st.markdown(
    "Tarea grupal - Maestria en Inteligencia Artificial Aplicada (UDLA). "
    "Esta app usa un clasificador **Haar Cascade** de OpenCV para detectar rostros, "
    "mostrando bounding boxes y FPS en tiempo real."
)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detectar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rostros = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    for (x, y, w, h) in rostros:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Rostro", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img, rostros


class DetectorRostros(VideoProcessorBase):
    """Procesador de video que aplica deteccion Haar Cascade fotograma a fotograma."""

    def __init__(self):
        self.tiempo_anterior = time.time()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img, rostros = detectar(img)

            tiempo_actual = time.time()
            fps = 1.0 / (tiempo_actual - self.tiempo_anterior) if tiempo_actual != self.tiempo_anterior else 0.0
            self.tiempo_anterior = tiempo_actual

            cv2.putText(img, "FPS: {:.1f}".format(fps), (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(img, "Rostros: {}".format(len(rostros)), (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            # Degradacion controlada: si algo falla en un fotograma, se devuelve el original
            return frame


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

tab_camara, tab_imagen = st.tabs(["Camara en vivo", "Subir imagen (respaldo)"])

with tab_camara:
    st.markdown("Concede permiso de camara al navegador para iniciar la deteccion en vivo.")
    webrtc_streamer(
        key="deteccion-haar-cascade",
        video_processor_factory=DetectorRostros,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with tab_imagen:
    st.markdown(
        "Si tu navegador no permite acceso a la camara, sube una imagen para "
        "probar la deteccion de todas formas."
    )
    archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        bytes_img = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
        img_anotada, rostros = detectar(img.copy())
        st.image(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB),
                  caption="Rostros detectados: {}".format(len(rostros)))

st.markdown("---")
st.markdown(
    "**Integrantes:** ___________, ___________, ___________, ___________  \n"
    "**Curso:** Vision por Computador - Programa de IA Aplicada, UDLA"
)
