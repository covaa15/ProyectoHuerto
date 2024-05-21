import pandas as pd
import requests
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap

st.set_page_config(layout="wide")  # Ajustar el diseño de la página

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
    .block-container {
        padding: 1rem 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título centrado
st.markdown("<h1 class='centered-title'>Huerto Inteligente 4.0</h1>", unsafe_allow_html=True)

# Definir colores para día y noche
colores = ListedColormap(['orange', 'lightblue'])

def obtenerDatos():
    try:
        fechaInicio = datetime(2024, 3, 8)  
        fechaActual = datetime.now()
        timestampInicio = int(fechaInicio.timestamp()) * 1000
        timestampActual = int(fechaActual.timestamp()) * 1000
        url = "https://sensecap.seeed.cc/openapi/list_telemetry_data"

        dispositivos = ['2CF7F1C0523000A2', '2CF7F1C05230009C', '2CF7F1C05230001D', '2CF7F1C043500730']
        
        # Diccionarios para almacenar los datos
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

        # Convertir diccionarios a DataFrames
        df_temperatura = pd.DataFrame(list(datos_temperatura.items()), columns=["Fecha", "Temperatura"])
        df_humedad = pd.DataFrame(list(datos_humedad.items()), columns=["Fecha", "Humedad"])
        df_conductibilidad = pd.DataFrame(list(datos_conductibilidad.items()), columns=["Fecha", "Conductibilidad"])

        return df_temperatura, df_humedad, df_conductibilidad

    except Exception as e:
        print(f"Error: {e}")

def crear_dataframe(df_temperatura, df_humedad, df_conductibilidad):
    # Unir los DataFrames en uno solo por la columna 'Fecha'
    df = pd.merge(df_temperatura, df_humedad, on="Fecha", how="outer")
    df = pd.merge(df, df_conductibilidad, on="Fecha", how="outer")

    # Convertir la columna 'Fecha' a tipo datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    # Resamplear los datos por hora
    df = df.set_index("Fecha").resample("H").mean().reset_index()

    # Verificar estructura del DataFrame
    print("Estructura del DataFrame:")
    print(df.info())
    # Ver los primeros registros del DataFrame
    print("\nPrimeros registros del DataFrame:")
    print(df.head(10))
    return df

def obtener_estado(fecha):
    try:
        horas_sol = {
            "Enero": [("08:55", "17:57"), ("08:55", "17:58"), ("08:55", "17:59"), ("08:55", "18:00"), ("08:55", "18:01"), ("08:55", "18:02"), ("08:55", "18:03"), ("08:55", "18:03"), ("08:55", "18:04"), ("08:55", "18:04"), ("07:59", "18:14"), ("08:36", "17:48")],
            "Febrero": [("08:55", "17:58"), ("08:38", "18:35"), ("07:56", "19:13"), ("08:02", "20:51"), ("07:14", "21:26"), ("06:44", "21:58"), ("06:47", "22:07"), ("07:14", "21:44"), ("07:49", "20:56"), ("08:22", "20:02"), ("08:00", "18:13"), ("08:37", "17:48")],
            "Marzo": [("08:55", "17:59"), ("08:37", "18:36"), ("07:55", "19:15"), ("08:00", "20:52"), ("07:12", "21:27"), ("06:44", "21:58"), ("06:48", "22:07"), ("07:16", "21:43"), ("07:50", "20:54"), ("08:23", "20:00"), ("08:01", "18:12"), ("08:38", "17:48")],
            "Abril": [("08:55", "18:00"), ("08:36", "18:38"), ("07:53", "19:16"), ("07:58", "20:53"), ("07:11", "21:28"), ("06:44", "21:59"), ("06:48", "22:07"), ("07:17", "21:41"), ("07:51", "20:53"), ("08:24", "19:58"), ("08:03", "18:10"), ("08:39", "17:47")]
        }

        fecha = pd.to_datetime(fecha)
        hora = fecha.strftime("%H:%M")

        for mes, horas in horas_sol.items():
            for hora_amanecer, hora_atardecer in horas:
                if hora_amanecer <= hora <= hora_atardecer:
                    return 1
        return 0
    except Exception as e:
        print(f"Error al obtener estado: {e}")

if __name__ == "__main__":
    # Llamada a la función para obtener datos y crear el DataFrame
    df_temperatura, df_humedad, df_conductibilidad = obtenerDatos()

    # Llamada a la función para crear el DataFrame combinado
    df = crear_dataframe(df_temperatura, df_humedad, df_conductibilidad)

    # Obtener el estado (día/noche) para cada registro
    df["Estado"] = df["Fecha"].apply(obtener_estado)

    # Separar las características (X) y la variable objetivo (y)
    X = df[["Temperatura", "Humedad", "Conductibilidad"]]
    y = df["Estado"]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un pipeline con un imputer y un modelo de regresión logística
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Reemplazar valores faltantes con la media
        ('scaler', StandardScaler()),                # Estandarizar características
        ('classifier', LogisticRegression())         # Modelo de regresión logística
    ])

    # Entrenar el pipeline en los datos de entrenamiento
    pipeline.fit(X_train, y_train)

    # Mostrar el selector de fecha en Streamlit
    fecha_seleccionada = st.date_input("Selecciona una fecha", value=datetime.now(), min_value=datetime(2024, 3, 28), max_value=datetime.now())
    
    # Filtrar los datos para la fecha seleccionada
    fecha_seleccionada_str = fecha_seleccionada.strftime('%Y-%m-%d')
    df_seleccionado = df[df["Fecha"].dt.strftime('%Y-%m-%d') == fecha_seleccionada_str]

    if not df_seleccionado.empty:
        # Obtener las predicciones para el día seleccionado
        X_seleccionado = df_seleccionado.drop(columns=["Fecha", "Estado"])
        predicciones = pipeline.predict(X_seleccionado)
        df_seleccionado["Predicciones"] = predicciones
        
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
    else:
        st.write("No hay datos disponibles para la fecha seleccionada.")
