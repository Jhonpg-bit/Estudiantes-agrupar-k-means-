from flask import Flask, render_template
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from sklearn.cluster import KMeans

# Inicializar la aplicación Flask
server = Flask(__name__)

# Cargar los datos desde el archivo CSV (Asegúrate de tener el archivo CSV con estudiantes)
df = pd.read_csv('/home/LimoncitoAgrio/mysite/dataset.csv')

# Convertir la asistencia a valores numéricos (A=1, F=0)
df['asistencia'] = df['asistencia'].map({'A': 1, 'F': 0})

# Crear una nueva columna para mostrar "A" o "F" en lugar de 1 y 0
df['asistencia_estudiante'] = df['asistencia'].map({1: 'A', 0: 'F'})

# Inicializar la aplicación Dash con estilos CSS externos
app_dash = Dash(__name__, server=server, url_base_pathname='/dash/', external_stylesheets=['/static/styles.css'])

# Layout de la aplicación Dash
app_dash.layout = html.Div([
    html.H1("Agrupación de Estudiantes (K-Means)", className="container"),

    # Campos para ingresar un nuevo estudiante
    html.Div([
        html.Label("Nombre del Estudiante:", className="container"),
        dcc.Input(id='nombre', type='text', value='', className="container"),

        html.Label("Rendimiento Académico (0-20):", className="container"),
        dcc.Input(id='rendimiento', type='number', min=0, max=20, value=10, className="container"),

        html.Label("Asistencia:", className="container"),
        dcc.Dropdown(id='asistencia', options=[
            {'label': 'Asistió (A)', 'value': 1},
            {'label': 'Faltó (F)', 'value': 0}
        ], value=1, className="container"),

        html.Label("Comportamiento (0-20):", className="container"),
        dcc.Input(id='comportamiento', type='number', min=0, max=20, value=10, className="container"),

        # Botón para agregar el nuevo estudiante
        html.Button('Agregar Estudiante', id='agregar_estudiante', n_clicks=0, className="container"),
    ], className="container"),

    # Gráfico interactivo
    dcc.Graph(id='grafico_estudiantes', className="graph-container"),

    # Mensaje de estado
    html.Div(id='mensaje', className="container")
])

# Callback para actualizar el gráfico y agregar el nuevo estudiante
@app_dash.callback(
    Output('grafico_estudiantes', 'figure'),
    Output('mensaje', 'children'),
    Input('agregar_estudiante', 'n_clicks'),
    State('nombre', 'value'),
    State('rendimiento', 'value'),
    State('asistencia', 'value'),
    State('comportamiento', 'value')
)
def actualizar_grafico(n_clicks, nombre, rendimiento, asistencia, comportamiento):
    global df

    # Si se presiona el botón para agregar un nuevo estudiante
    if n_clicks > 0 and nombre:
        # Crear un nuevo DataFrame para el estudiante
        nuevo_estudiante = pd.DataFrame({
            'nombre': [nombre],
            'rendimiento': [rendimiento],
            'asistencia': [asistencia],
            'asistencia_estudiante': ['A' if asistencia == 1 else 'F'],
            'comportamiento': [comportamiento]
        })

        # Concatenar el nuevo estudiante al DataFrame
        df = pd.concat([df, nuevo_estudiante], ignore_index=True)

        # Volver a entrenar el modelo KMeans
        X = df[['rendimiento', 'asistencia', 'comportamiento']].astype(float)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)

        # Añadir las etiquetas de cluster al DataFrame
        df['cluster'] = kmeans.labels_

        # Crear la figura
        fig = px.scatter(
            df,
            x='rendimiento',
            y='comportamiento',
            color='cluster',
            hover_name='nombre',
            hover_data={'asistencia_estudiante': True},
            title='Agrupación de Estudiantes (Rendimiento vs Comportamiento)',
            labels={'rendimiento': 'Rendimiento Académico (0-20)', 'comportamiento': 'Comportamiento (0-20)'}
        )

        fig.add_scatter(
            x=[rendimiento],
            y=[comportamiento],
            mode='markers',
            marker=dict(color='red', size=12, symbol='diamond'),
            name=f'Nuevo estudiante: {nombre}'
        )

        mensaje = f'Estudiante {nombre} agregado correctamente al cluster.'
        return fig, mensaje

    # Si no se presionó el botón, simplemente devolver el gráfico existente
    fig = px.scatter(
        df,
        x='rendimiento',
        y='comportamiento',
        color='cluster',
        hover_name='nombre',
        hover_data={'asistencia_estudiante': True},
        title='Agrupación de Estudiantes (Rendimiento vs Comportamiento)',
        labels={'rendimiento': 'Rendimiento Académico (0-20)', 'comportamiento': 'Comportamiento (0-20)'}
    )

    return fig, "Agrega un nuevo estudiante usando el formulario."


# Ruta principal en Flask
@server.route('/')
def home():
    return render_template('index.html')


# Ejecutar la aplicación
if __name__ == '__main__':
    server.run(debug=True)
