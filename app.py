import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Habitaciones Madrid",
    page_icon="📊",
    layout="wide"
)

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('habitaciones_madrid.csv')
    # Convertir columnas numéricas
    numeric_cols = ['price', 'size', 'rooms', 'bathrooms', 'latitude', 'longitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calcular precio por metro cuadrado
    df['price_per_m2'] = df['price'] / df['size']
    
    # Crear segmentos de tamaño
    size_bins = [0, 10, 15, 20, 25, 30, 40, 50, 100, float('inf')]
    size_labels = ['<10m²', '10-15m²', '15-20m²', '20-25m²', '25-30m²', '30-40m²', '40-50m²', '50-100m²', '>100m²']
    df['size_segment'] = pd.cut(df['size'], bins=size_bins, labels=size_labels)
    
    return df

# Título principal
st.title("📊 Análisis Macro del Mercado de Habitaciones en Madrid")
st.write("Este dashboard proporciona un análisis general del mercado de habitaciones en Madrid, mostrando métricas clave y tendencias.")

# Cargar datos
df = load_data()

# Sidebar para filtros
st.sidebar.header("Filtros para el Análisis")

# Filtro de distritos
district_options = sorted(df['district'].dropna().unique())
selected_districts = st.sidebar.multiselect(
    "Distritos a incluir en el análisis",
    options=district_options,
    default=district_options
)

# Filtro de barrios
neighborhood_options = sorted(df['neighborhood'].dropna().unique())
selected_neighborhoods = st.sidebar.multiselect(
    "Barrios a incluir en el análisis",
    options=neighborhood_options,
    default=[]
)

# Filtro de rango de precios
min_price = int(df['price'].min())
max_price = int(df['price'].max())
price_range = st.sidebar.slider(
    "Rango de precio (€/mes)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# Filtro de segmentos de tamaño
size_segment_options = df['size_segment'].dropna().unique().tolist()
selected_size_segments = st.sidebar.multiselect(
    "Segmentos de tamaño",
    options=size_segment_options,
    default=size_segment_options
)

# Aplicar filtros
filtered_df = df.copy()

# Filtro de distritos
if selected_districts:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]

# Filtro de barrios
if selected_neighborhoods:
    filtered_df = filtered_df[filtered_df['neighborhood'].isin(selected_neighborhoods)]

# Filtro de precio
filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) &
                          (filtered_df['price'] <= price_range[1])]

# Filtro de segmentos de tamaño
if selected_size_segments:
    filtered_df = filtered_df[filtered_df['size_segment'].isin(selected_size_segments)]

# Verificar si hay datos después de filtrar
if len(filtered_df) == 0:
    st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
    st.stop()

# Métricas generales
st.header("Métricas Generales del Mercado")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Número de Habitaciones", f"{len(filtered_df)}")

with col2:
    avg_price = round(filtered_df['price'].mean(), 2)
    st.metric("Precio Promedio", f"{avg_price} €/mes")

with col3:
    avg_size = round(filtered_df['size'].mean(), 2)
    st.metric("Tamaño Promedio", f"{avg_size} m²")

with col4:
    avg_price_per_m2 = round(filtered_df['price_per_m2'].mean(), 2)
    st.metric("Precio Promedio por m²", f"{avg_price_per_m2} €/m²")

# Análisis por Segmento de Tamaño
st.header("Análisis por Segmento de Tamaño")

# Calcular métricas por segmento de tamaño
size_segment_metrics = filtered_df.groupby('size_segment').agg(
    habitaciones=('propertyCode', 'count'),
    precio_promedio=('price', 'mean'),
    precio_min=('price', 'min'),
    precio_max=('price', 'max'),
    precio_por_m2=('price_per_m2', 'mean')
).reset_index()

# Ordenar por segmento de tamaño (para mantener el orden lógico)
size_segment_metrics['size_segment'] = pd.Categorical(
    size_segment_metrics['size_segment'], 
    categories=size_segment_options,
    ordered=True
)
size_segment_metrics = size_segment_metrics.sort_values('size_segment')

# Mostrar tabla de métricas por segmento de tamaño
st.dataframe(
    size_segment_metrics.style.format({
        'precio_promedio': '{:.2f} €',
        'precio_min': '{:.2f} €',
        'precio_max': '{:.2f} €',
        'precio_por_m2': '{:.2f} €/m²'
    }),
    use_container_width=True
)

# Visualización de segmentos de tamaño
col1, col2 = st.columns(2)

with col1:
    # Gráfico de distribución de habitaciones por segmento de tamaño
    fig = px.bar(
        size_segment_metrics,
        x='size_segment',
        y='habitaciones',
        title='Distribución de Habitaciones por Segmento de Tamaño',
        labels={'size_segment': 'Segmento de Tamaño', 'habitaciones': 'Número de Habitaciones'},
        color='habitaciones',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Gráfico de precio promedio por segmento de tamaño
    fig = px.bar(
        size_segment_metrics,
        x='size_segment',
        y='precio_promedio',
        title='Precio Promedio por Segmento de Tamaño',
        labels={'size_segment': 'Segmento de Tamaño', 'precio_promedio': 'Precio Promedio (€/mes)'},
        color='precio_promedio',
        color_continuous_scale='Viridis',
        text_auto='.2f'
    )
    st.plotly_chart(fig, use_container_width=True)

# Gráfico de precio por m² por segmento de tamaño
fig = px.bar(
    size_segment_metrics,
    x='size_segment',
    y='precio_por_m2',
    title='Precio por m² por Segmento de Tamaño',
    labels={'size_segment': 'Segmento de Tamaño', 'precio_por_m2': 'Precio por m² (€/m²)'},
    color='precio_por_m2',
    color_continuous_scale='Viridis',
    text_auto='.2f'
)
st.plotly_chart(fig, use_container_width=True)

# Análisis por Distrito
st.header("Análisis por Distrito")

# Calcular métricas por distrito
district_metrics = filtered_df.groupby('district').agg(
    habitaciones=('propertyCode', 'count'),
    precio_promedio=('price', 'mean'),
    tamaño_promedio=('size', 'mean'),
    precio_por_m2=('price_per_m2', 'mean')
).reset_index()

# Ordenar por número de habitaciones
district_metrics = district_metrics.sort_values('habitaciones', ascending=False)

# Mostrar tabla de métricas por distrito
st.dataframe(
    district_metrics.style.format({
        'precio_promedio': '{:.2f} €',
        'tamaño_promedio': '{:.2f} m²',
        'precio_por_m2': '{:.2f} €/m²'
    }),
    use_container_width=True
)

# Análisis Cruzado: Distrito vs Segmento de Tamaño
st.header("Análisis Cruzado: Distrito vs Segmento de Tamaño")

# Crear pivot table para el análisis cruzado
pivot_size_district = filtered_df.pivot_table(
    values='price',
    index='district',
    columns='size_segment',
    aggfunc='mean'
).fillna(0)

# Mostrar mapa de calor
fig = px.imshow(
    pivot_size_district,
    title='Precio Promedio por Distrito y Segmento de Tamaño',
    labels=dict(x="Segmento de Tamaño", y="Distrito", color="Precio Promedio (€)"),
    color_continuous_scale='Viridis',
    text_auto='.0f'
)
st.plotly_chart(fig, use_container_width=True)

# Visualizaciones
st.header("Visualizaciones Adicionales")

tab1, tab2, tab3, tab4 = st.tabs(["Precios por Distrito", "Tamaño por Distrito", "Distribución de Precios", "Mapa de Calor"])

with tab1:
    # Gráfico de precios promedio por distrito
    fig = px.bar(
        district_metrics.sort_values('precio_promedio', ascending=False),
        x='district',
        y='precio_promedio',
        title='Precio Promedio por Distrito',
        labels={'district': 'Distrito', 'precio_promedio': 'Precio Promedio (€/mes)'},
        color='precio_promedio',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Gráfico de tamaño promedio por distrito
    fig = px.bar(
        district_metrics.sort_values('tamaño_promedio', ascending=False),
        x='district',
        y='tamaño_promedio',
        title='Tamaño Promedio por Distrito',
        labels={'district': 'Distrito', 'tamaño_promedio': 'Tamaño Promedio (m²)'},
        color='tamaño_promedio',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Histograma de distribución de precios
    fig = px.histogram(
        filtered_df,
        x='price',
        nbins=30,
        title='Distribución de Precios',
        labels={'price': 'Precio (€/mes)', 'count': 'Número de Habitaciones'},
        color_discrete_sequence=['#3366CC']
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Mapa de calor de precio por distrito y número de habitaciones
    pivot_data = filtered_df.pivot_table(
        values='price',
        index='district',
        columns='rooms',
        aggfunc='mean'
    ).fillna(0)

    fig = px.imshow(
        pivot_data,
        title='Precio Promedio por Distrito y Número de Habitaciones',
        labels=dict(x="Número de Habitaciones", y="Distrito", color="Precio Promedio (€)"),
        color_continuous_scale='Viridis',
        text_auto='.0f'
    )
    st.plotly_chart(fig, use_container_width=True)

# Análisis de Correlación
st.header("Análisis de Correlación")

# Seleccionar columnas numéricas para la correlación
numeric_cols = ['price', 'size', 'rooms', 'bathrooms', 'price_per_m2']
corr_df = filtered_df[numeric_cols].corr()

# Crear mapa de calor de correlación
fig = px.imshow(
    corr_df,
    text_auto=True,
    title='Matriz de Correlación entre Variables Numéricas',
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1
)
st.plotly_chart(fig, use_container_width=True)

# Análisis de Dispersión
st.header("Relación entre Variables")

# Gráfico de dispersión: Precio vs Tamaño, coloreado por segmento de tamaño
fig = px.scatter(
    filtered_df,
    x='size',
    y='price',
    color='size_segment',
    title='Relación entre Precio y Tamaño',
    labels={'size': 'Tamaño (m²)', 'price': 'Precio (€/mes)', 'size_segment': 'Segmento de Tamaño'},
    hover_data=['address', 'neighborhood', 'district'],
    opacity=0.7
)
st.plotly_chart(fig, use_container_width=True)

# Análisis Geoespacial
st.header("Análisis Geoespacial")

# Filtrar datos para el mapa
map_data = filtered_df[['latitude', 'longitude', 'price', 'district', 'size_segment']].dropna()

# Crear mapa de precios
fig = px.scatter_mapbox(
    map_data,
    lat='latitude',
    lon='longitude',
    color='size_segment',
    size='price',
    size_max=15,
    zoom=11,
    title='Distribución Geográfica por Segmento de Tamaño',
    mapbox_style="carto-positron",
    hover_data=['district', 'price', 'size_segment']
)
st.plotly_chart(fig, use_container_width=True)

# Análisis por Barrio (Top 10)
st.header("Top 10 Barrios por Precio Promedio")

# Calcular métricas por barrio
neighborhood_metrics = filtered_df.groupby('neighborhood').agg(
    habitaciones=('propertyCode', 'count'),
    precio_promedio=('price', 'mean'),
    tamaño_promedio=('size', 'mean'),
    precio_por_m2=('price_per_m2', 'mean')
).reset_index()

# Filtrar barrios con al menos 5 habitaciones
neighborhood_metrics = neighborhood_metrics[neighborhood_metrics['habitaciones'] >= 5]

# Top 10 barrios más caros
top_expensive = neighborhood_metrics.sort_values('precio_promedio', ascending=False).head(10)

fig = px.bar(
    top_expensive,
    x='neighborhood',
    y='precio_promedio',
    title='Top 10 Barrios más Caros',
    labels={'neighborhood': 'Barrio', 'precio_promedio': 'Precio Promedio (€/mes)'},
    color='precio_promedio',
    color_continuous_scale='Viridis',
    text_auto='.2f'
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Conclusiones
st.header("Conclusiones del Análisis")

# Calcular algunas estadísticas para las conclusiones
most_expensive_district = district_metrics.loc[district_metrics['precio_promedio'].idxmax(), 'district']
most_expensive_price = district_metrics.loc[district_metrics['precio_promedio'].idxmax(), 'precio_promedio']

largest_size_district = district_metrics.loc[district_metrics['tamaño_promedio'].idxmax(), 'district']
largest_size = district_metrics.loc[district_metrics['tamaño_promedio'].idxmax(), 'tamaño_promedio']

best_value_district = district_metrics.loc[district_metrics['precio_por_m2'].idxmin(), 'district']
best_value_price = district_metrics.loc[district_metrics['precio_por_m2'].idxmin(), 'precio_por_m2']

# Análisis por segmento de tamaño
most_common_segment = filtered_df['size_segment'].value_counts().idxmax()
most_expensive_segment = size_segment_metrics.loc[size_segment_metrics['precio_promedio'].idxmax(), 'size_segment']
most_expensive_segment_price = size_segment_metrics.loc[size_segment_metrics['precio_promedio'].idxmax(), 'precio_promedio']
best_value_segment = size_segment_metrics.loc[size_segment_metrics['precio_por_m2'].idxmin(), 'size_segment']
best_value_segment_price = size_segment_metrics.loc[size_segment_metrics['precio_por_m2'].idxmin(), 'precio_por_m2']

st.write(f"""
### Conclusiones por Distrito:
- El distrito más caro es **{most_expensive_district}** con un precio promedio de **{most_expensive_price:.2f}€/mes**.
- El distrito con habitaciones más grandes es **{largest_size_district}** con un tamaño promedio de **{largest_size:.2f}m²**.
- El distrito con mejor relación precio/tamaño es **{best_value_district}** con un precio por m² de **{best_value_price:.2f}€/m²**.

### Conclusiones por Segmento de Tamaño:
- El segmento de tamaño más común es **{most_common_segment}**.
- El segmento de tamaño más caro es **{most_expensive_segment}** con un precio promedio de **{most_expensive_segment_price:.2f}€/mes**.
- El segmento con mejor relación precio/tamaño es **{best_value_segment}** con un precio por m² de **{best_value_segment_price:.2f}€/m²**.

### Correlaciones:
- La correlación entre precio y tamaño es de **{corr_df.loc['price', 'size']:.2f}**, lo que indica una relación {abs(corr_df.loc['price', 'size']) > 0.5 and 'fuerte' or 'moderada a débil'} entre estas variables.
""")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
Esta aplicación proporciona un análisis macro del mercado de habitaciones en Madrid.
Utiliza los filtros para personalizar el análisis según tus intereses.
""")
