import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Cargar el modelo y los datos
modelo = joblib.load('mejor_modelo_dia_noche.pkl')
df = joblib.load('datos_huerto.pkl')

# Configuración de la página
st.set_page_config(layout="wide", page_title="Huerto Inteligente 4.0", page_icon="💻")

# CSS para el diseño personalizado
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #00FFFF, #007FFF, #0000FF);
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .centered-title {
        text-align: center;
        color: white;
        font-size: 3rem;
        margin-bottom: 20px;
    }
    .subheader {
        color: white;
        font-size: 1.5rem;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .dataframe-container, .metric-container, .summary-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #D3D3D3;
        padding: 10px;
        margin-bottom: 20px;
    }
    .metric {
        font-size: 1.2rem;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título de la aplicación
st.markdown('<h1 class="centered-title">Huerto Inteligente 4.0</h1>', unsafe_allow_html=True)

# Selectores y tabla de datos
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="subheader">Datos de Predicción</div>', unsafe_allow_html=True)
    años_disponibles1 = df['Fecha'].dt.year.unique()
    año_seleccionado1 = st.selectbox("Selecciona un año (Tabla)", años_disponibles1, key="year_table")

    meses_disponibles1 = df[df['Fecha'].dt.year == año_seleccionado1]['Fecha'].dt.month.unique()
    mes_seleccionado1 = st.selectbox("Selecciona un mes (Tabla)", meses_disponibles1, key="month_table")

    df_filtrado = df[(df['Fecha'].dt.year == año_seleccionado1) & (df['Fecha'].dt.month == mes_seleccionado1)]

    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df_filtrado)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="subheader">Seleccionar Fecha</div>', unsafe_allow_html=True)
    fecha_seleccionada = st.date_input("Selecciona una fecha", value=datetime.now(), min_value=datetime(2024, 3, 28), max_value=datetime.now())

    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Métricas de Evaluación del modelo</div>', unsafe_allow_html=True)

    # Resumen del día seleccionado
    fecha_seleccionada_str = fecha_seleccionada.strftime('%Y-%m-%d')
    df_seleccionado = df[df["Fecha"].dt.strftime('%Y-%m-%d') == fecha_seleccionada_str]

    if not df_seleccionado.empty:
        y_verdadero = df_seleccionado["Estado"]
        X_seleccionado = df_seleccionado.drop(columns=["Fecha", "Estado"])
        predicciones = modelo.predict(X_seleccionado)
        df_seleccionado["Predicciones"] = predicciones

        accuracy = accuracy_score(y_verdadero, predicciones)
        precision = precision_score(y_verdadero, predicciones, average='binary')
        puntuacion = f1_score(y_verdadero, predicciones)

        st.markdown(f'<div class="metric">Precisión (Accuracy): {accuracy:.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">Precisión: {precision:.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">Puntuación del modelo: {puntuacion:.2f}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric">No hay métricas disponibles para la fecha seleccionada.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="summary-container">', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Resumen del Día Seleccionado</div>', unsafe_allow_html=True)

    if not df_seleccionado.empty:
        temp_media = df_seleccionado['Temperatura'].mean()
        humedad_media = df_seleccionado['Humedad'].mean()
        conductividad_media = df_seleccionado['Conductibilidad'].mean()

        st.markdown(f'<div class="metric">Temperatura Media del Suelo: {temp_media:.2f} °C</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">Humedad Media del Suelo: {humedad_media:.2f} %</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric">Conductividad Media: {conductividad_media:.2f} μS/cm</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric">No hay datos disponibles para la fecha seleccionada.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Gráfica de las Mediciones de un Día
if not df_seleccionado.empty:
    st.markdown('<div class="subheader">Gráfica de las Mediciones de un Día</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)

    columnas = ["Temperatura", "Humedad", "Conductibilidad"]
    colores_grafica = ['red', 'blue', 'green']
    titulos = ["Temperatura del Suelo", "Humedad del Suelo", "Conductibilidad"]

    for i, ax in enumerate(axes):
        ax.plot(df_seleccionado["Fecha"], df_seleccionado[columnas[i]], label=columnas[i], color=colores_grafica[i])

        for j in range(len(df_seleccionado) - 1):
            color = 'orange' if df_seleccionado["Predicciones"].iloc[j] == 0 else 'lightblue'
            ax.axvspan(df_seleccionado["Fecha"].iloc[j], df_seleccionado["Fecha"].iloc[j+1], color=color, alpha=0.3)

        ax.set_ylabel(columnas[i])
        ax.legend()
        ax.set_title(f"{titulos[i]} del {fecha_seleccionada_str}")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel("Hora")

    handles = [plt.Line2D([0], [0], color='orange', lw=4, label='Noche'),
               plt.Line2D([0], [0], color='lightblue', lw=4, label='Día')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    plt.tight_layout()

    st.pyplot(fig)
else:
    st.write("No hay datos disponibles para la fecha seleccionada.")

# Selectores de año y mes para gráficas
st.markdown('<div class="subheader">Generar Gráficas de X días</div>', unsafe_allow_html=True)
años_disponibles = df['Fecha'].dt.year.unique()
año_seleccionado = st.selectbox("Selecciona un año (Gráficas)", años_disponibles, key="year_plots")

meses_disponibles = df[df['Fecha'].dt.year == año_seleccionado]['Fecha'].dt.month.unique()
mes_seleccionado = st.selectbox("Selecciona un mes (Gráficas)", meses_disponibles, key="month_plots")

df_filtrado = df[(df['Fecha'].dt.year == año_seleccionado) & (df['Fecha'].dt.month == mes_seleccionado)]

show_date_selector = False
if 'df_filtrado' in locals() and not df_filtrado.empty:
    show_date_selector = True
    fechas_seleccionadas = st.multiselect("Selecciona varios días", pd.date_range(start=df_filtrado['Fecha'].min(), end=df_filtrado['Fecha'].max(), freq='D'))

if st.button("Generar gráficas"):
    if show_date_selector and len(fechas_seleccionadas) > 0:
        fechas_seleccionadas_str = [fecha.strftime('%d-%m-%Y') for fecha in fechas_seleccionadas]
        df_seleccionado = df_filtrado[df_filtrado["Fecha"].dt.strftime('%d-%m-%Y').isin(fechas_seleccionadas_str)]

        if not df_seleccionado.empty:
            df_seleccionado = df_seleccionado.sort_values(by='Fecha')

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i, (columna, color, titulo) in enumerate(zip(["Temperatura", "Humedad", "Conductibilidad"], ['red', 'blue', 'green'], ["Temperatura del Suelo por Día", "Humedad del Suelo por Día", "Conductibilidad por Día"])):
                axs[i].plot(df_seleccionado["Fecha"], df_seleccionado[columna], color=color)
                axs[i].set_xlabel("Fecha")
                axs[i].set_ylabel(columna)
                axs[i].set_title(titulo)
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
                plt.setp(axs[i].get_xticklabels(), rotation=45)

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No hay datos disponibles para las fechas seleccionadas.")
    else:
        if not show_date_selector:
            st.write("Por favor selecciona un año y mes primero.")
        else:
            st.write("Por favor selecciona al menos un día.")
