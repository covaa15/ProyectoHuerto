import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, f1_score
import requests
from collections import defaultdict
from dateutil import parser
import tkinter as tk
from tkinter import ttk, messagebox


# Cargar el modelo y los datos
modelo = joblib.load('mejor_modelo_dia_noche.pkl')
df = joblib.load('datos_huerto.pkl')

def obtenerDatos():
    try:
        fechaInicio = datetime(2024, 3, 8)
        fechaActual = datetime.now()
        timestampInicio = int(fechaInicio.timestamp()) * 1000
        timestampActual = int(fechaActual.timestamp()) * 1000
        url = "https://sensecap.seeed.cc/openapi/list_telemetry_data"

        dispositivos = ['2CF7F1C0523000A2', '2CF7F1C05230009C', '2CF7F1C05230001D', '2CF7F1C043500730']

        datos_temperatura = defaultdict(list)
        datos_humedad = defaultdict(list)
        datos_conductibilidad = defaultdict(list)

        for device_eui in dispositivos:
            params = {
                'device_eui':  device_eui,
                'channel_index': 1,
                "time_start": str(timestampInicio),
                "time_end": str(timestampActual)
            }
            respuesta = requests.get(url, params=params, auth=('93I2S5UCP1ISEF4F', '6552EBDADED14014B18359DB4C3B6D4B3984D0781C2545B6A33727A4BBA1E46E'))

            if respuesta.status_code == 200:
                datos = respuesta.json()
                tipoMedicion = list()
                bloque0 = list()
                bloque1 = list()
                bloque2 = list()
                indice = 0  

                for i, lista in enumerate(datos["data"]["list"]):
                    for medicion in lista:
                        if(i == 0):
                            tipoMedicion.append(medicion[1])
                        else:
                            if indice == 0:
                                bloque0.append(medicion)
                            elif indice == 1:
                                bloque1.append(medicion)
                            else:
                                bloque2.append(medicion)
                            indice = indice + 1

                posicion4102 = next((i for i, x in enumerate(tipoMedicion) if x == "4102"), None)
                posicion4103 = next((i for i, x in enumerate(tipoMedicion) if x == "4103"), None)
                posicion4108 = next((i for i, x in enumerate(tipoMedicion) if x == "4108"), None)

                medicionTemperatura = bloque0[0] if posicion4102 == 0 else (
                    bloque1[0] if posicion4102 == 1 else bloque2[0])

                medicionHumedad = bloque0[0] if posicion4103 == 0 else (
                    bloque1[0] if posicion4103 == 1 else bloque2[0])

                medicionC02 = bloque0[0] if posicion4108 == 0 else (
                    bloque1[0] if posicion4108 == 1 else bloque2[0])
                
                for med_temp in medicionTemperatura:
                    fecha = parser.parse(med_temp[1]).strftime('%Y-%m-%d %H:%M:%S')
                    datos_temperatura[fecha]= med_temp[0]

                for med_temp in medicionHumedad:
                    fecha = parser.parse(med_temp[1]).strftime('%Y-%m-%d %H:%M:%S')
                    datos_humedad[fecha]= med_temp[0]

                for med_temp in medicionC02:
                    fecha = parser.parse(med_temp[1]).strftime('%Y-%m-%d %H:%M:%S')
                    datos_conductibilidad[fecha]= med_temp[0]

        df_temperatura = pd.DataFrame(list(datos_temperatura.items()), columns=["Fecha", "Temperatura"])
        df_humedad = pd.DataFrame(list(datos_humedad.items()), columns=["Fecha", "Humedad"])
        df_conductibilidad = pd.DataFrame(list(datos_conductibilidad.items()), columns=["Fecha", "Conductibilidad"])

        return df_temperatura, df_humedad, df_conductibilidad

    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Función para crear el DataFrame combinado
def crear_dataframe(df_temperatura, df_humedad, df_conductibilidad):
    df = pd.merge(df_temperatura, df_humedad, on="Fecha", how="outer")
    df = pd.merge(df, df_conductibilidad, on="Fecha", how="outer")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.set_index("Fecha").resample("H").mean().reset_index()
    return df

# Función para determinar si es de día o de noche
def obtener_estado(fecha):
    try:
        horas_sol = {
           "Enero": [("08:55", "17:57"), ("08:55", "17:58"), ("08:55", "17:59"), ("08:55", "18:00"), ("08:55", "18:01"), ("08:55", "18:02"), ("08:55", "18:03"), ("08:55", "18:04"), ("08:55", "18:05"), ("08:54", "18:06"), ("08:54", "18:07"), ("08:54", "18:08"), ("08:53", "18:09"), ("08:53", "18:11"), ("08:52", "18:12"), ("08:52", "18:13"), ("08:51", "18:14"), ("08:51", "18:15"), ("08:50", "18:17"), ("08:50", "18:18"), ("08:49", "18:19"), ("08:48", "18:20"), ("08:47", "18:22"), ("08:47", "18:23"), ("08:46", "18:24"), ("08:45", "18:26"), ("08:44", "18:27"), ("08:43", "18:28"), ("08:42", "18:30"), ("08:41", "18:31"), ("08:40", "18:32")],
           "Febrero": [("08:39", "18:34"), ("08:38", "18:35"), ("08:37", "18:36"), ("08:36", "18:38"), ("08:35", "18:39"), ("08:33", "18:40"), ("08:32", "18:42"), ("08:31", "18:43"), ("08:30", "18:45"), ("08:28", "18:46"), ("08:27", "18:47"), ("08:26", "18:49"), ("08:24", "18:50"), ("08:23", "18:51"), ("08:21", "18:53"), ("08:20", "18:54"), ("08:19", "18:55"), ("08:17", "18:57"), ("08:16", "18:58"), ("08:14", "18:59"), ("08:13", "19:01"), ("08:11", "19:02"), ("08:09", "19:03"), ("08:08", "19:04"), ("08:06", "19:06"), ("08:05", "19:07"), ("08:03", "19:08"), ("08:01", "19:10"), ("08:00", "19:11")],
           "Marzo": [("07:58", "19:12"), ("07:56", "19:13"), ("07:55", "19:15"), ("07:53", "19:16"), ("07:51", "19:17"), ("07:50", "19:18"), ("07:48", "19:20"), ("07:46", "19:21"), ("07:44", "19:22"), ("07:43", "19:23"), ("07:41", "19:25"), ("07:39", "19:26"), ("07:37", "19:27"), ("07:36", "18:28"), ("07:34", "19:29"), ("07:32", "19:31"), ("07:30", "19:32"), ("07:28", "19:33"), ("07:27", "19:34"), ("07:25", "19:35"), ("07:23", "19:37"), ("07:21", "19:38"), ("07:20", "19:39"), ("07:18", "19:40"), ("07:16", "19:41"), ("07:14", "19:43"), ("07:12", "19:44"), ("07:11", "19:45"), ("07:09", "19:46"), ("07:07", "19:47"), ("08:05", "20:49")],
           "Abril": [("08:03", "20:50"), ("08:02", "20:51"), ("08:00", "20:52"), ("07:58", "20:53"), ("07:56", "20:54"), ("07:55", "20:56"), ("07:53", "20:57"), ("07:51", "20:58"), ("07:50", "20:59"), ("07:48", "21:00"), ("07:46", "21:01"), ("07:44", "21:03"), ("07:43", "21:04"), ("07:41", "21:05"), ("07:39", "21:06"), ("07:38", "21:07"), ("07:36", "21:09"), ("07:35", "21:10"), ("07:33", "21:11"), ("07:31", "21:12"), ("07:30", "21:13"), ("07:28", "21:14"), ("07:27", "21:16"), ("07:25", "21:17"), ("07:24", "21:18"), ("07:22", "21:19"), ("07:21", "21:20"), ("07:19", "21:21"), ("07:18", "21:23"), ("07:16", "21:24")],
           "Mayo": [("07:15", "21:25"), ("07:14", "21:26"), ("07:12", "21:27"), ("07:11", "21:28"), ("07:10", "21:30"), ("07:08", "21:31"), ("07:07", "21:32"), ("07:06", "21:33"), ("07:05", "21:34"), ("07:03", "21:35"), ("07:02", "21:36"), ("07:01", "21:37"), ("07:00", "21:38"), ("06:59", "21:40"), ("06:58", "21:41"), ("06:57", "21:42"), ("06:56", "21:43"), ("06:55", "21:44"), ("06:54", "21:45"), ("06:53", "21:46"), ("06:52", "21:47"), ("06:51", "21:48"), ("06:51", "21:49"), ("06:50", "21:50"), ("06:49", "21:51"), ("06:48", "21:52"), ("06:48", "21:53"), ("06:47", "21:53"), ("06:46", "21:54"), ("06:46", "21:55"), ("06:45", "21:56")],
           "Junio": [("06:45", "21:57"), ("06:44", "21:58"), ("06:44", "21:58"), ("06:44", "21:59"), ("06:43", "22:00"), ("06:43", "22:00"), ("06:43", "22:01"), ("06:42", "22:02"), ("06:42", "22:02"), ("06:42", "22:03"), ("06:42", "22:03"), ("06:42", "22:04"), ("06:42", "22:04"), ("06:42", "22:05"), ("06:42", "22:05"), ("06:42", "22:06"), ("06:42", "22:06"), ("06:42", "22:06"), ("06:42", "22:07"), ("06:42", "22:07"), ("06:43", "22:07"), ("06:43", "22:07"), ("06:43", "22:07"), ("06:43", "22:07"), ("06:44", "22:08"), ("06:44", "22:08"), ("06:45", "22:08"), ("06:45", "22:08"), ("06:46", "22:07"), ("06:46", "22:07")],
           "Julio": [("06:47", "22:07"), ("06:47", "22:07"), ("06:48", "22:07"), ("06:48", "22:07"), ("06:49", "22:06"), ("06:50", "22:06"), ("06:50", "22:05"), ("06:51", "22:05"), ("06:52", "22:05"), ("06:53", "22:04"), ("06:53", "22:04"), ("06:54", "22:03"), ("06:55", "22:02"), ("06:56", "22:02"), ("06:57", "22:01"), ("06:58", "22:00"), ("06:58", "22:00"), ("06:59", "21:59"), ("07:00", "21:58"), ("07:01", "21:57"), ("07:02", "21:56"), ("07:03", "21:56"), ("07:04", "21:55"), ("07:05", "21:54"), ("07:06", "21:53"), ("07:07", "21:52"), ("07:08", "21:51"), ("07:09", "21:50"), ("07:10", "21:48"), ("07:11", "21:47"), ("07:12", "21:46")],
           "Agosto": [("07:13", "21:45"), ("07:14", "21:44"), ("07:16", "21:43"), ("07:17", "21:41"), ("07:18", "21:40"), ("07:19", "21:39"), ("07:20", "21:37"), ("07:21", "21:36"), ("07:22", "21:35"), ("07:23", "21:33"), ("07:24", "21:32"), ("07:25", "21:30"), ("07:27", "21:29"), ("07:28", "21:27"), ("07:29", "21:26"), ("07:30", "21:24"), ("07:31", "21:23"), ("07:32", "21:21"), ("07:33", "21:20"), ("07:34", "21:18"), ("07:35", "21:16"), ("07:36", "21:15"), ("07:38", "21:13"), ("07:39", "21:12"), ("07:40", "21:10"), ("07:41", "21:08"), ("07:42", "21:07"), ("07:43", "21:05"), ("07:44", "21:03"), ("07:45", "21:01"), ("07:46", "21:00")],
           "Septiembre": [("07:48", "20:58"), ("07:49", "20:56"), ("07:50", "20:54"), ("07:51", "20:53"), ("07:52", "20:51"), ("07:53", "20:49"), ("07:54", "20:47"), ("07:55", "20:46"), ("07:56", "20:44"), ("07:57", "20:42"), ("07:59", "20:40"), ("08:00", "20:38"), ("08:01", "20:36"), ("08:02", "20:35"), ("08:03", "20:33"), ("08:04", "20:31"), ("08:05", "20:29"), ("08:06", "20:27"), ("08:07", "20:26"), ("08:09", "20:24"), ("08:10", "20:22"), ("08:11", "20:20"), ("08:12", "20:18"), ("08:13", "20:16"), ("08:14", "20:15"), ("08:15", "20:13"), ("08:16", "20:11"), ("08:17", "20:09"), ("08:19", "20:07"), ("08:20", "20:66")],
           "Octubre": [("08:21", "20:04"), ("08:22", "20:02"), ("08:23", "20:00"), ("08:24", "19:58"), ("08:26", "19:57"), ("08:27", "19:55"), ("08:28", "19:53"), ("08:29", "19:51"), ("08:30", "19:50"), ("08:31", "19:48"), ("08:33", "19:46"), ("08:34", "19:45"), ("08:35", "19:43"), ("08:36", "19:41"), ("08:37", "19:40"), ("08:39", "19:38"), ("08:40", "19:36"), ("08:41", "19:35"), ("08:42", "19:33"), ("08:44", "19:32"), ("08:45", "19:30"), ("08:46", "19:28"), ("08:47", "19:27"), ("08:49", "19:25"), ("08:50", "19:24"), ("08:51", "19:23"), ("07:52", "18:21"), ("07:54", "18:20"), ("07:55", "18:18"), ("07:56", "18:17"), ("07:57", "18:15")],
           "Noviembre": [("07:59", "18:14"), ("08:00", "18:13"), ("08:01", "18:12"), ("08:03", "18:10"), ("08:04", "18:09"), ("08:05", "18:08"), ("08:07", "18:07"), ("08:08", "18:06"), ("08:09", "18:04"), ("08:10", "18:03"), ("08:12", "18:02"), ("08:13", "18:01"), ("08:14", "18:00"), ("08:16", "17:59"), ("08:17", "17:58"), ("08:18", "17:57"), ("08:19", "17:57"), ("08:21", "17:56"), ("08:22", "17:55"), ("08:23", "17:54"), ("08:24", "17:53"), ("08:26", "17:53"), ("08:27", "17:52"), ("08:28", "17:51"), ("08:29", "17:51"), ("08:30", "17:50"), ("08:32", "17:50"), ("08:33", "17:49"),("08:34", "17:49"), ("08:35", "17:49") ],
           "Diciembre": [("08:36", "17:48"), ("08:37", "17:48"), ("08:38", "17:48"), ("08:39", "17:47"), ("08:40", "17:47"), ("08:41", "17:47"), ("08:42", "17:47"), ("08:43", "17:47"), ("08:44", "17:47"), ("08:45", "17:47"), ("08:46", "17:47"),("08:46", "17:47"), ("08:47", "17:47"), ("08:48", "17:48"), ("08:49", "17:48"), ("08:49", "17:48"), ("08:50", "17:48"), ("08:51", "17:49"), ("08:51", "17:49"), ("08:52", "17:50"), ("08:52", "17:50"), ("08:53", "17:51"), ("08:53", "17:51"), ("08:54", "17:52"), ("08:54", "17:52"), ("08:54", "17:53"), ("08:55", "17:54"), ("08:55", "17:55"), ("08:55", "17:55"), ("08:55", "17:56"), ("08:55", "17:57")]
           }

        meses_espanol = {
            'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo', 'April': 'Abril',
            'May': 'Mayo', 'June': 'Junio', 'July': 'Julio', 'August': 'Agosto',
            'September': 'Septiembre', 'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
        }

        fecha_dt = pd.to_datetime(fecha)
        mes_nombre = fecha_dt.strftime('%B')
        dia = fecha_dt.day
        
        mes_nombre_espanol = meses_espanol[mes_nombre]
        
        hora = fecha_dt.strftime('%H:%M')

        hora_salida, hora_puesta = horas_sol[mes_nombre_espanol][dia - 1]

        estado = 1 if hora_salida <= hora <= hora_puesta else 0
        return estado
    except Exception as e:
        return f"Error al determinar el estado: {e}"

# Función para cargar el modelo y los datos
def cargar_datos():
    df_temperatura, df_humedad, df_conductibilidad = obtenerDatos()
    df = crear_dataframe(df_temperatura, df_humedad, df_conductibilidad)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Estado'] = df['Fecha'].apply(lambda x: obtener_estado(x.strftime('%Y-%m-%d %H:%M:%S')))
    df = df.dropna()
    return df



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
        font-size: 1.6rem;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .dataframe-container, .metric-container, .summary-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #000000;
        padding: 10px;
        margin-bottom: 20px;
    }
    .metric {
        font-size:  1.4rem;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título de la aplicación
st.markdown('<h1 class="centered-title">Huerto Inteligente 4.0</h1>', unsafe_allow_html=True)

st.markdown("""
    <style>
    .custom-expander .streamlit-expanderHeader {
        border: 2px solid black; /* Cambia el grosor y color del borde aquí */
	    color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


with st.expander("Normas de Uso del Programa"):
    st.markdown("""
    <div style='color: #FFFFFF;'>
    1. <strong>Actualización de Datos:</strong>
        - El botón de 'Actualizar' carga los datos desde la última actualización hasta el momento actual. Este proceso tarda aproximadamente un minuto. Es recomendable usar esta función para obtener los datos más recientes.
    
    2. <strong>Selección de Año y Mes:</strong>
        - En la columna izquierda, selecciona un año y un mes. A continuación, la tabla cargará los datos predichos por el modelo para todos los días disponibles en ese año y mes.
    
    3. <strong>Gráficas Diarias:</strong>
        - En la columna derecha, selecciona una fecha específica. Para esa fecha, se generarán tres tipos de gráficas: temperatura del suelo, humedad del suelo y conductibilidad.
    
    4. <strong>Generación de Gráficas Mensuales:</strong>
        - Para crear gráficas mensuales, primero selecciona un año y un mes. Se cargarán todos los días disponibles para esa selección. Luego, elige los días específicos para los cuales deseas generar las gráficas y pulsa el botón 'Generar Gráficas'.
    </div>
    """, unsafe_allow_html=True)

# Esto es para actualizar el dataframe que consulta los datos.
# Tardará unos minutos pero luego muestra los datos completamente actualizados.
st.markdown('<div class="subheader">Actualizar datos</div>', unsafe_allow_html=True)
if st.button("Actualizar datos"):
    df = cargar_datos()
    # Esto además guarda la última obtención de datos para tener siempre una "copia de seguridad" actualizada.
    # En caso de que el servidor se reiniciase, se cargaría esta copia y solo se actualizarían los datos
    # ...en caso de que se pulse el botón. Esto permite consultar datos viejos desde la última actualización.
    joblib.dump(df, 'datos_huerto.pkl')
    

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
st.markdown('<div class="subheader">Generar Gráficas Mensuales</div>', unsafe_allow_html=True)
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
