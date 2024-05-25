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
st.set_page_config(layout="wide")

# CSS para centrar el título
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    .main {
        padding: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Título de la aplicación
st.markdown('<h1 class="centered-title">Huerto Inteligente 4.0</h1>', unsafe_allow_html=True)


# Mostrar la tabla de datos
st.subheader("Predicción de Día/Noche realizada por el modelo")
# Obtener los años únicos disponibles en los datos
años_disponibles1 = df['Fecha'].dt.year.unique()
año_seleccionado1 = st.selectbox("Selecciona un año (Tabla)", años_disponibles1, key="year_table")

# Obtener los meses únicos disponibles en los datos filtrados
meses_disponibles1 = df[df['Fecha'].dt.year == año_seleccionado1]['Fecha'].dt.month.unique()
mes_seleccionado1 = st.selectbox("Selecciona un mes (Tabla)", meses_disponibles1, key="month_table")

# Filtrar los datos por año y mes seleccionado
df_filtrado = df[(df['Fecha'].dt.year == año_seleccionado1) & (df['Fecha'].dt.month == mes_seleccionado1)]

# Mostrar los datos filtrados
st.dataframe(df_filtrado)



# Mostrar el selector de fecha en Streamlit
fecha_seleccionada = st.date_input("Selecciona una fecha", value=datetime.now(), min_value=datetime(2024, 3, 28), max_value=datetime.now())

# Filtrar los datos para la fecha seleccionada
fecha_seleccionada_str = fecha_seleccionada.strftime('%Y-%m-%d')
df_seleccionado = df[df["Fecha"].dt.strftime('%Y-%m-%d') == fecha_seleccionada_str]

if not df_seleccionado.empty:
    # Obtener las predicciones para el día seleccionado
    y_verdadero = df_seleccionado["Estado"]
    X_seleccionado = df_seleccionado.drop(columns=["Fecha", "Estado"])
    predicciones = modelo.predict(X_seleccionado)
    df_seleccionado["Predicciones"] = predicciones
    # Calcular las métricas de evaluación
    accuracy = accuracy_score(y_verdadero, predicciones)
    precision = precision_score(y_verdadero, predicciones, average='binary')
    puntuacion=f1_score(y_verdadero, predicciones)

        
    # Crear las gráficas
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)

    # Lista de columnas y colores
    columnas = ["Temperatura", "Humedad", "Conductibilidad"]
    colores_grafica = ['red', 'blue', 'green']
    titulos = ["Temperatura", "Humedad", "Conductibilidad"]

    for i, ax in enumerate(axes):
        ax.plot(df_seleccionado["Fecha"], df_seleccionado[columnas[i]], label=columnas[i], color=colores_grafica[i])

        # Colorear los tramos de día y noche
        for j in range(len(df_seleccionado) - 1):
            color = 'orange' if df_seleccionado["Predicciones"].iloc[j] == 0 else 'lightblue'
            ax.axvspan(df_seleccionado["Fecha"].iloc[j], df_seleccionado["Fecha"].iloc[j+1], color=color, alpha=0.3)
        
        ax.set_ylabel(columnas[i])
        ax.legend()
        ax.set_title(f"{titulos[i]} del {fecha_seleccionada_str}")

    # Formatear la gráfica
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel("Hora")

    # Añadir leyenda común para los colores
    handles = [plt.Line2D([0], [0], color='orange', lw=4, label='Noche'),
               plt.Line2D([0], [0], color='lightblue', lw=4, label='Día')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    # Ajustar los espacios entre los subplots
    plt.tight_layout()

    # Mostrar las gráficas en Streamlit
    st.pyplot(fig)

    st.subheader("Métricas de Evaluación del modelo")
    st.write(f"Precisión Accuracy: {accuracy}")
    st.write(f"Precisión: {precision}")
    st.write(f"Puntuación del modelo{puntuacion}")

else:
    st.write("No hay datos disponibles para la fecha seleccionada.")

# Obtener los años únicos disponibles en los datos
años_disponibles = df['Fecha'].dt.year.unique()
año_seleccionado = st.selectbox("Selecciona un año (Gráficas)", años_disponibles, key="year_plots")

# Obtener los meses únicos disponibles en los datos filtrados
meses_disponibles = df[df['Fecha'].dt.year == año_seleccionado]['Fecha'].dt.month.unique()
mes_seleccionado = st.selectbox("Selecciona un mes (Gráficas)", meses_disponibles, key="month_plots")

# Filtrar los datos por año y mes seleccionado
df_filtrado = df[(df['Fecha'].dt.year == año_seleccionado) & (df['Fecha'].dt.month == mes_seleccionado)]

# Mostrar el selector de fecha en Streamlit para seleccionar varios días
show_date_selector = False
if 'df_filtrado' in locals() and not df_filtrado.empty:
    show_date_selector = True
    fechas_seleccionadas = st.multiselect("Selecciona varios días", pd.date_range(start=df_filtrado['Fecha'].min(), end=df_filtrado['Fecha'].max(), freq='D'))

# Agregar un botón para generar las gráficas
if st.button("Generar gráficas"):
    if show_date_selector and len(fechas_seleccionadas) > 0:
        # Filtrar los datos para las fechas seleccionadas
        fechas_seleccionadas_str = [fecha.strftime('%d-%m-%Y') for fecha in fechas_seleccionadas]
        df_seleccionado = df_filtrado[df_filtrado["Fecha"].dt.strftime('%d-%m-%Y').isin(fechas_seleccionadas_str)]

        if not df_seleccionado.empty:
            # Ordenar los datos por fecha
            df_seleccionado = df_seleccionado.sort_values(by='Fecha')

            # Crear una figura y ejes para las tres gráficas
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            # Crear las gráficas para cada tipo de medición
            for i, (columna, color, titulo) in enumerate(zip(["Temperatura", "Humedad", "Conductibilidad"], ['red', 'blue', 'green'], ["Temperatura por día", "Humedad por día", "Conductibilidad por día"])):
                axs[i].plot(df_seleccionado["Fecha"], df_seleccionado[columna], color=color)
                axs[i].set_xlabel("Fecha")
                axs[i].set_ylabel(columna)
                axs[i].set_title(titulo)
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
                plt.setp(axs[i].get_xticklabels(), rotation=45)  # Rotar etiquetas del eje x

            # Ajustar el espacio entre las gráficas
            plt.tight_layout()

            # Mostrar las gráficas en Streamlit
            st.pyplot(fig)
        else:
            st.write("No hay datos disponibles para las fechas seleccionadas.")
    else:
        if not show_date_selector:
            st.write("Por favor selecciona un año y mes primero.")
        else:
            st.write("Por favor selecciona al menos un día.")
