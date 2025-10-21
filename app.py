import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import plotly.express as px


# -----------------------------
# Cargar el modelo
# -----------------------------
model = load_model("keras_model.h5")
class_names = ["Adriana", "Wendy", "Desconocido"]

# -----------------------------
# Base de datos SQLite
# -----------------------------
conn = sqlite3.connect('predicciones.db', check_same_thread=False)
c = conn.cursor()

# Crear tabla predicciones
c.execute('''
CREATE TABLE IF NOT EXISTS predicciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT,
    fuente TEXT,
    etiqueta TEXT,
    confianza REAL
)
''')

# Crear tabla usuarios
c.execute('''
CREATE TABLE IF NOT EXISTS usuario (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    apellido TEXT,
    correo TEXT,
    telefono TEXT,
    nacimiento TEXT
)
''')
conn.commit()

# -----------------------------
# Funciones para guardar datos
# -----------------------------
def guardar_prediccion(fuente, etiqueta, confianza):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO predicciones (fecha, fuente, etiqueta, confianza) VALUES (?, ?, ?, ?)',
              (fecha, fuente, etiqueta, confianza))
    conn.commit()

def guardar_datos(nombre, apellido, correo, telefono, nacimiento):
    c.execute('INSERT INTO usuario (nombre, apellido, correo, telefono, nacimiento) VALUES (?, ?, ?, ?, ?)',
              (nombre, apellido, correo, telefono, nacimiento))
    conn.commit()

# Ejemplo de guardar usuarios iniciales
guardar_datos("Adriana", "Cornejo", "adri@gmail.com", "987654321", "2005-03-13")
guardar_datos("Wendy", "Llivichuzca", "wendy@gmail.com", "986535241", "2005-05-11")

# -----------------------------
# Interfaz Streamlit
# -----------------------------
st.title("Reconocimiento de Imágenes - Teachable Machine")
menu = ["En Vivo", "Administración", "Analítica"]
choice = st.sidebar.selectbox("Secciones", menu)

# -----------------------------
# Sección En Vivo
# -----------------------------
if choice == "En Vivo":
    st.header("Clasificación en Tiempo Real (Webcam)")

    cam_image = st.camera_input("Toma una foto")  # Sin loop ni key duplicada

    if cam_image is not None:
        img = Image.open(cam_image).convert('RGB')
        st.image(img, caption='Imagen capturada', use_column_width=True)

        # Preprocesar imagen
        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Predicción
        pred = model.predict(x)
        clase_idx = np.argmax(pred, axis=1)[0]
        clase_nombre = class_names[clase_idx]
        confianza = float(pred[0][clase_idx])

        st.success(f"Predicción: {clase_nombre} ({confianza*100:.2f}%)")

        # Buscar usuario
        c.execute('SELECT * FROM usuario WHERE nombre=?', (clase_nombre,))
        usuario = c.fetchone()
        if usuario:
            st.info(f"Usuario reconocido: {usuario[1]} {usuario[2]}, correo: {usuario[3]}")
        else:
            st.warning("Usuario no registrado en la base de datos")

        # Guardar predicción
        guardar_prediccion('camara', clase_nombre, confianza)


# -----------------------------
# Sección Administración
# -----------------------------
elif choice == "Administración":
    st.header("Administración de Predicciones")
    df = pd.read_sql_query("SELECT * FROM predicciones", conn)
    st.dataframe(df)

    if st.button("Exportar CSV"):
        df.to_csv("predicciones.csv", index=False)
        st.success("CSV generado: predicciones.csv")
    st.download_button(
    label="📥 Descargar CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="compras_filtradas.csv",
    mime="text/csv"
)

# -----------------------------
# Sección Analítica
# -----------------------------
elif choice == "Analítica":
    st.header("Panel Analítico")
    df = pd.read_sql_query("SELECT * FROM predicciones", conn)

    if not df.empty:
        # Crear columna hora
        df['hora'] = pd.to_datetime(df['fecha']).dt.hour

        # Gráfica 1: Distribución de etiquetas (Dona)
        etiqueta_counts = df['etiqueta'].value_counts().reset_index()
        etiqueta_counts.columns = ['etiqueta', 'conteo']
        fig1 = px.pie(etiqueta_counts, values='conteo', names='etiqueta', hole=0.4,
                      title="Distribución de etiquetas", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig1, use_container_width=True)

        # Gráfica 2: Confianza por etiqueta (Bubble)
        df['confianza_scaled'] = df['confianza']*100
        fig2 = px.scatter(df, x='etiqueta', y='confianza', size='confianza_scaled',
                          color='etiqueta', hover_data=['fecha', 'fuente'],
                          title="Confianza por etiqueta (Bubble Chart)", size_max=60,
                          color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig2, use_container_width=True)

        # Gráfica 3: Predicciones por fuente (Dona)
        fuente_counts = df['fuente'].value_counts().reset_index()
        fuente_counts.columns = ['fuente', 'conteo']
        fig3 = px.pie(fuente_counts, values='conteo', names='fuente', hole=0.3,
                      title="Predicciones por fuente", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)

        # Gráfica 4: Confianza promedio por etiqueta (Barras)
        df_mean = df.groupby('etiqueta')['confianza'].mean().reset_index()
        fig4 = px.bar(df_mean, x='etiqueta', y='confianza', color='etiqueta',
                      title="Confianza promedio por etiqueta", text=df_mean['confianza'],
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig4.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)

        # Gráfica 5: Predicciones por hora (Barras)
        hora_counts = df['hora'].value_counts().sort_index().reset_index()
        hora_counts.columns = ['hora', 'count']
        fig5 = px.bar(hora_counts, x='hora', y='count',
                      labels={'hora':'Hora', 'count':'Cantidad'},
                      title="Predicciones por hora",
                      text='count',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig5, use_container_width=True)

        # Gráfica 6: Confianza vs Fuente y Etiqueta (Bubble interactiva)
        fig6 = px.scatter(df, x='fuente', y='confianza', size='confianza', color='etiqueta',
                          hover_data=['fecha'], size_max=50,
                          title="Confianza vs Fuente por etiqueta (Bubble)")
        st.plotly_chart(fig6, use_container_width=True)
