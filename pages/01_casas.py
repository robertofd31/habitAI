
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Rentabilidad - Propiedades Madrid",
    page_icon="💰",
    layout="wide"
)

# Título principal
st.title("💰 Análisis de Rentabilidad de Propiedades en Madrid")
st.write("Este dashboard analiza la rentabilidad potencial de propiedades en venta si se alquilan por habitaciones.")


# Cargar los datos
@st.cache_data
def load_property_data():
    df = pd.read_csv('propiedades_madrid_hasta_200k.csv')
    # Convertir columnas numéricas
    numeric_cols = ['price', 'size', 'rooms', 'bathrooms', 'latitude', 'longitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Asegurar que rooms sea al menos 1 (para estudios)
    df['rooms'] = df['rooms'].fillna(1)
    df.loc[df['rooms'] == 0, 'rooms'] = 1

    # Calcular precio por metro cuadrado
    df['price_per_m2'] = df['price'] / df['size']

    return df

@st.cache_data
def load_room_data():
    # Cargar datos de habitaciones en alquiler
    df_rooms = pd.read_csv('habitaciones_madrid.csv')
    # Convertir columnas numéricas
    numeric_cols = ['price', 'size']
    for col in numeric_cols:
        if col in df_rooms.columns:
            df_rooms[col] = pd.to_numeric(df_rooms[col], errors='coerce')

    return df_rooms

# Cargar datos
df_properties = load_property_data()
df_rooms = load_room_data()

# Sidebar para filtros
st.sidebar.header("Filtros para el Análisis")

# Filtro de distritos
district_options = sorted(df_properties['district'].dropna().unique())
selected_districts = st.sidebar.multiselect(
    "Distritos a incluir en el análisis",
    options=district_options,
    default=district_options[:5]  # Seleccionar los primeros 5 por defecto
)

# Filtro de barrios
neighborhood_options = sorted(df_properties['neighborhood'].dropna().unique())
selected_neighborhoods = st.sidebar.multiselect(
    "Barrios a incluir en el análisis",
    options=neighborhood_options,
    default=[]
)

# Filtro de rango de precios
min_price = int(df_properties['price'].min())
max_price = int(df_properties['price'].max())
price_range = st.sidebar.slider(
    "Rango de precio de compra (€)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# Filtro de tamaño
min_size = int(df_properties['size'].min())
max_size = int(df_properties['size'].max())
size_range = st.sidebar.slider(
    "Rango de tamaño (m²)",
    min_value=min_size,
    max_value=max_size,
    value=(min_size, max_size)
)

# Filtro de número de habitaciones
room_options = sorted(df_properties['rooms'].dropna().unique())
selected_rooms = st.sidebar.multiselect(
    "Número de habitaciones",
    options=room_options,
    default=room_options
)

# Parámetros para el cálculo de rentabilidad
st.sidebar.header("Parámetros de Rentabilidad")

# Precio medio de alquiler por habitación (por distrito)
avg_room_price_by_district = df_rooms.groupby('district')['price'].mean().to_dict()
default_room_price = int(df_rooms['price'].mean())

# Precio de alquiler tradicional (estimado como % del valor de compra anualizado)
traditional_rental_yield = st.sidebar.slider(
    "Rentabilidad anual alquiler tradicional (%)",
    min_value=2.0,
    max_value=8.0,
    value=4.0,
    step=0.1
)

# Gastos de reforma por m²
renovation_cost_per_m2 = st.sidebar.slider(
    "Coste de reforma por m² (€)",
    min_value=0,
    max_value=1000,
    value=300,
    step=50
)

# Coste de añadir una habitación (tabiques, puertas, etc.)
cost_per_added_room = st.sidebar.slider(
    "Coste por habitación adicional (€)",
    min_value=1000,
    max_value=10000,
    value=3000,
    step=500
)

# Gastos mensuales (comunidad, IBI, seguros, etc.)
monthly_expenses_percent = st.sidebar.slider(
    "Gastos mensuales (% del alquiler)",
    min_value=5.0,
    max_value=30.0,
    value=15.0,
    step=1.0
)

# Tasa de ocupación
occupancy_rate = st.sidebar.slider(
    "Tasa de ocupación (%)",
    min_value=50.0,
    max_value=100.0,
    value=90.0,
    step=5.0
)

# Aplicar filtros
filtered_df = df_properties.copy()

# Filtro de distritos
if selected_districts:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]

# Filtro de barrios
if selected_neighborhoods:
    filtered_df = filtered_df[filtered_df['neighborhood'].isin(selected_neighborhoods)]

# Filtro de precio
filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) &
                          (filtered_df['price'] <= price_range[1])]

# Filtro de tamaño
filtered_df = filtered_df[(filtered_df['size'] >= size_range[0]) &
                          (filtered_df['size'] <= size_range[1])]

# Filtro de habitaciones
if selected_rooms:
    filtered_df = filtered_df[filtered_df['rooms'].isin(selected_rooms)]

# Verificar si hay datos después de filtrar
if len(filtered_df) == 0:
    st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
    st.stop()

# Función para calcular la rentabilidad
def calculate_roi(row, added_rooms=0, district_avg_prices=avg_room_price_by_district):
    district = row['district']

    # Precio medio de habitación en el distrito o valor por defecto
    avg_room_price = district_avg_prices.get(district, default_room_price)

    # Número total de habitaciones (originales + añadidas)
    total_rooms = row['rooms'] + added_rooms

    # Verificar si el tamaño permite añadir habitaciones (mínimo 10m² por habitación)
    if row['size'] / total_rooms < 10:
        total_rooms = max(1, int(row['size'] / 10))  # Al menos una habitación

    # Ingresos mensuales por alquiler de habitaciones
    monthly_income_rooms = total_rooms * avg_room_price * (occupancy_rate / 100)

    # Ingresos mensuales por alquiler tradicional
    monthly_income_traditional = (row['price'] * (traditional_rental_yield / 100)) / 12

    # Gastos mensuales
    monthly_expenses = monthly_income_rooms * (monthly_expenses_percent / 100)

    # Beneficio neto mensual
    net_monthly_profit_rooms = monthly_income_rooms - monthly_expenses
    net_monthly_profit_traditional = monthly_income_traditional - (monthly_income_traditional * (monthly_expenses_percent / 100))

    # Inversión inicial
    initial_investment = row['price']

    # Coste de reforma
    renovation_cost = row['size'] * renovation_cost_per_m2

    # Coste de añadir habitaciones
    room_addition_cost = added_rooms * cost_per_added_room

    # Inversión total
    total_investment = initial_investment + renovation_cost + room_addition_cost

    # ROI anual (%)
    annual_roi_rooms = (net_monthly_profit_rooms * 12 / total_investment) * 100
    annual_roi_traditional = (net_monthly_profit_traditional * 12 / initial_investment) * 100

    # Tiempo de recuperación de la inversión (años)
    payback_period_rooms = total_investment / (net_monthly_profit_rooms * 12) if net_monthly_profit_rooms > 0 else float('inf')
    payback_period_traditional = initial_investment / (net_monthly_profit_traditional * 12) if net_monthly_profit_traditional > 0 else float('inf')

    return {
        'total_rooms': total_rooms,
        'monthly_income_rooms': monthly_income_rooms,
        'monthly_income_traditional': monthly_income_traditional,
        'net_monthly_profit_rooms': net_monthly_profit_rooms,
        'net_monthly_profit_traditional': net_monthly_profit_traditional,
        'total_investment': total_investment,
        'annual_roi_rooms': annual_roi_rooms,
        'annual_roi_traditional': annual_roi_traditional,
        'payback_period_rooms': payback_period_rooms,
        'payback_period_traditional': payback_period_traditional
    }

# Calcular rentabilidad para cada propiedad
results = []
for _, row in filtered_df.iterrows():
    # Escenario actual (sin añadir habitaciones)
    base_roi = calculate_roi(row)
    base_roi['property_id'] = row['propertyCode']
    base_roi['district'] = row['district']
    base_roi['neighborhood'] = row['neighborhood']
    base_roi['original_price'] = row['price']
    base_roi['size'] = row['size']
    base_roi['original_rooms'] = row['rooms']
    base_roi['added_rooms'] = 0
    results.append(base_roi)

    # Escenario añadiendo 1 habitación
    roi_1_room = calculate_roi(row, added_rooms=1)
    roi_1_room['property_id'] = row['propertyCode']
    roi_1_room['district'] = row['district']
    roi_1_room['neighborhood'] = row['neighborhood']
    roi_1_room['original_price'] = row['price']
    roi_1_room['size'] = row['size']
    roi_1_room['original_rooms'] = row['rooms']
    roi_1_room['added_rooms'] = 1
    results.append(roi_1_room)

    # Escenario añadiendo 2 habitaciones
    roi_2_rooms = calculate_roi(row, added_rooms=2)
    roi_2_rooms['property_id'] = row['propertyCode']
    roi_2_rooms['district'] = row['district']
    roi_2_rooms['neighborhood'] = row['neighborhood']
    roi_2_rooms['original_price'] = row['price']
    roi_2_rooms['size'] = row['size']
    roi_2_rooms['original_rooms'] = row['rooms']
    roi_2_rooms['added_rooms'] = 2
    results.append(roi_2_rooms)

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# Métricas generales
st.header("Métricas Generales de Rentabilidad")

col1, col2, col3 = st.columns(3)

with col1:
    avg_roi_rooms = results_df[results_df['added_rooms'] == 0]['annual_roi_rooms'].mean()
    st.metric("ROI Promedio (Alquiler por Habitaciones)", f"{avg_roi_rooms:.2f}%")

with col2:
    avg_roi_traditional = results_df[results_df['added_rooms'] == 0]['annual_roi_traditional'].mean()
    st.metric("ROI Promedio (Alquiler Tradicional)", f"{avg_roi_traditional:.2f}%")

with col3:
    roi_difference = avg_roi_rooms - avg_roi_traditional
    st.metric("Diferencia de ROI", f"{roi_difference:.2f}%", delta=f"{roi_difference:.2f}%")

# Análisis por Distrito
st.header("Análisis de Rentabilidad por Distrito")

# Calcular métricas por distrito
district_metrics = results_df[results_df['added_rooms'] == 0].groupby('district').agg(
    num_properties=('property_id', 'count'),
    avg_roi_rooms=('annual_roi_rooms', 'mean'),
    avg_roi_traditional=('annual_roi_traditional', 'mean'),
    avg_payback_rooms=('payback_period_rooms', 'mean'),
    avg_payback_traditional=('payback_period_traditional', 'mean')
).reset_index()

# Calcular diferencia de ROI
district_metrics['roi_difference'] = district_metrics['avg_roi_rooms'] - district_metrics['avg_roi_traditional']

# Ordenar por diferencia de ROI
district_metrics = district_metrics.sort_values('roi_difference', ascending=False)

# Mostrar tabla de métricas por distrito
st.dataframe(
    district_metrics.style.format({
        'avg_roi_rooms': '{:.2f}%',
        'avg_roi_traditional': '{:.2f}%',
        'roi_difference': '{:.2f}%',
        'avg_payback_rooms': '{:.2f} años',
        'avg_payback_traditional': '{:.2f} años'
    }),
    use_container_width=True
)

# Visualización de ROI por distrito
fig = px.bar(
    district_metrics,
    x='district',
    y=['avg_roi_rooms', 'avg_roi_traditional'],
    barmode='group',
    title='ROI por Distrito: Alquiler por Habitaciones vs. Tradicional',
    labels={
        'district': 'Distrito',
        'value': 'ROI (%)',
        'variable': 'Tipo de Alquiler'
    },
    color_discrete_map={
        'avg_roi_rooms': '#1f77b4',
        'avg_roi_traditional': '#ff7f0e'
    }
)

fig.update_layout(
    xaxis_tickangle=-45,
    legend=dict(
        title='',
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)

# Análisis por Barrio
st.header("Análisis de Rentabilidad por Barrio")

# Calcular métricas por barrio
neighborhood_metrics = results_df[results_df['added_rooms'] == 0].groupby('neighborhood').agg(
    num_properties=('property_id', 'count'),
    avg_roi_rooms=('annual_roi_rooms', 'mean'),
    avg_roi_traditional=('annual_roi_traditional', 'mean'),
    avg_payback_rooms=('payback_period_rooms', 'mean'),
    avg_payback_traditional=('payback_period_traditional', 'mean')
).reset_index()

# Calcular diferencia de ROI
neighborhood_metrics['roi_difference'] = neighborhood_metrics['avg_roi_rooms'] - neighborhood_metrics['avg_roi_traditional']

# Filtrar barrios con al menos 3 propiedades
min_properties = st.slider("Mínimo de propiedades por barrio", 1, 20, 3)
filtered_neighborhoods = neighborhood_metrics[neighborhood_metrics['num_properties'] >= min_properties]

# Ordenar por diferencia de ROI
filtered_neighborhoods = filtered_neighborhoods.sort_values('roi_difference', ascending=False)

# Mostrar tabla de métricas por barrio
st.dataframe(
    filtered_neighborhoods.style.format({
        'avg_roi_rooms': '{:.2f}%',
        'avg_roi_traditional': '{:.2f}%',
        'roi_difference': '{:.2f}%',
        'avg_payback_rooms': '{:.2f} años',
        'avg_payback_traditional': '{:.2f} años'
    }),
    use_container_width=True
)

# Top 10 barrios con mayor ROI en alquiler por habitaciones
top_neighborhoods = filtered_neighborhoods.sort_values('avg_roi_rooms', ascending=False).head(10)

fig = px.bar(
    top_neighborhoods,
    x='neighborhood',
    y='avg_roi_rooms',
    title='Top 10 Barrios con Mayor ROI en Alquiler por Habitaciones',
    labels={
        'neighborhood': 'Barrio',
        'avg_roi_rooms': 'ROI (%)'
    },
    color='avg_roi_rooms',
    color_continuous_scale='Viridis',
    text_auto='.2f'
)

fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Análisis de Habitaciones Adicionales
st.header("Impacto de Añadir Habitaciones")

# Agrupar por número de habitaciones adicionales
added_rooms_metrics = results_df.groupby('added_rooms').agg(
    avg_roi_rooms=('annual_roi_rooms', 'mean'),
    avg_payback_rooms=('payback_period_rooms', 'mean'),
    avg_total_investment=('total_investment', 'mean')
).reset_index()

# Mostrar tabla de métricas por habitaciones adicionales
st.dataframe(
    added_rooms_metrics.style.format({
        'avg_roi_rooms': '{:.2f}%',
        'avg_payback_rooms': '{:.2f} años',
        'avg_total_investment': '{:,.2f} €'
    }),
    use_container_width=True
)

# Visualización del impacto de añadir habitaciones
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(
        x=added_rooms_metrics['added_rooms'],
        y=added_rooms_metrics['avg_roi_rooms'],
        name='ROI (%)',
        marker_color='#1f77b4'
    )
)

fig.add_trace(
    go.Scatter(
        x=added_rooms_metrics['added_rooms'],
        y=added_rooms_metrics['avg_payback_rooms'],
        name='Tiempo de Recuperación (años)',
        marker_color='#ff7f0e',
        mode='lines+markers'
    ),
    secondary_y=True
)

fig.update_layout(
    title='Impacto de Añadir Habitaciones en ROI y Tiempo de Recuperación',
    xaxis_title='Habitaciones Adicionales',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

fig.update_yaxes(title_text='ROI (%)', secondary_y=False)
fig.update_yaxes(title_text='Tiempo de Recuperación (años)', secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Análisis por Distrito y Habitaciones Adicionales
st.header("ROI por Distrito y Habitaciones Adicionales")

# Calcular métricas por distrito y habitaciones adicionales
district_rooms_metrics = results_df.groupby(['district', 'added_rooms']).agg(
    avg_roi_rooms=('annual_roi_rooms', 'mean')
).reset_index()

# Crear pivot table para el análisis cruzado
pivot_district_rooms = district_rooms_metrics.pivot(
    index='district',
    columns='added_rooms',
    values='avg_roi_rooms'
).fillna(0)

# Renombrar columnas
pivot_district_rooms.columns = [f'{int(col)} habitaciones adicionales' for col in pivot_district_rooms.columns]

# Mostrar mapa de calor
fig = px.imshow(
    pivot_district_rooms,
    title='ROI (%) por Distrito y Habitaciones Adicionales',
    labels=dict(x="Habitaciones Adicionales", y="Distrito", color="ROI (%)"),
    color_continuous_scale='Viridis',
    text_auto='.2f'
)

st.plotly_chart(fig, use_container_width=True)

# Propiedades con Mayor Potencial
st.header("Propiedades con Mayor Potencial de Rentabilidad")

# Filtrar propiedades con ROI positivo
positive_roi_properties = results_df[results_df['annual_roi_rooms'] > 0]

# Ordenar por ROI en alquiler por habitaciones
top_properties = positive_roi_properties.sort_values('annual_roi_rooms', ascending=False)

# Mostrar las mejores propiedades
st.subheader("Top Propiedades por ROI (Sin Habitaciones Adicionales)")
top_base_properties = top_properties[top_properties['added_rooms'] == 0].head(10)
st.dataframe(
    top_base_properties[['property_id', 'district', 'neighborhood', 'original_price', 'size', 'original_rooms', 'annual_roi_rooms', 'payback_period_rooms']].style.format({
        'original_price': '{:,.2f} €',
        'size': '{:.2f} m²',
        'annual_roi_rooms': '{:.2f}%',
        'payback_period_rooms': '{:.2f} años'
    }),
    use_container_width=True
)

st.subheader("Top Propiedades por ROI (Con 1 Habitación Adicional)")
top_1room_properties = top_properties[top_properties['added_rooms'] == 1].head(10)
st.dataframe(
    top_1room_properties[['property_id', 'district', 'neighborhood', 'original_price', 'size', 'original_rooms', 'total_rooms', 'annual_roi_rooms', 'payback_period_rooms']].style.format({
        'original_price': '{:,.2f} €',
        'size': '{:.2f} m²',
        'annual_roi_rooms': '{:.2f}%',
        'payback_period_rooms': '{:.2f} años'
    }),
    use_container_width=True
)

st.subheader("Top Propiedades por ROI (Con 2 Habitaciones Adicionales)")
top_2room_properties = top_properties[top_properties['added_rooms'] == 2].head(10)
st.dataframe(
    top_2room_properties[['property_id', 'district', 'neighborhood', 'original_price', 'size', 'original_rooms', 'total_rooms', 'annual_roi_rooms', 'payback_period_rooms']].style.format({
        'original_price': '{:,.2f} €',
        'size': '{:.2f} m²',
        'annual_roi_rooms': '{:.2f}%',
        'payback_period_rooms': '{:.2f} años'
    }),
    use_container_width=True
)

# Análisis de Tamaño vs. ROI
st.header("Relación entre Tamaño y ROI")

# Crear gráfico de dispersión
fig = px.scatter(
    results_df[results_df['added_rooms'] == 0],
    x='size',
    y='annual_roi_rooms',
    color='district',
    size='original_price',
    hover_data=['neighborhood', 'original_rooms', 'total_rooms', 'payback_period_rooms'],
    title='Relación entre Tamaño y ROI por Distrito',
    labels={
        'size': 'Tamaño (m²)',
        'annual_roi_rooms': 'ROI (%)',
        'district': 'Distrito',
        'original_price': 'Precio (€)'
    }
)

st.plotly_chart(fig, use_container_width=True)

# Conclusiones
st.header("Conclusiones del Análisis")

# Calcular algunas estadísticas para las conclusiones
best_district = district_metrics.loc[district_metrics['avg_roi_rooms'].idxmax(), 'district']
best_district_roi = district_metrics.loc[district_metrics['avg_roi_rooms'].idxmax(), 'avg_roi_rooms']

best_neighborhood = filtered_neighborhoods.loc[filtered_neighborhoods['avg_roi_rooms'].idxmax(), 'neighborhood']
best_neighborhood_roi = filtered_neighborhoods.loc[filtered_neighborhoods['avg_roi_rooms'].idxmax(), 'avg_roi_rooms']

best_added_rooms = added_rooms_metrics.loc[added_rooms_metrics['avg_roi_rooms'].idxmax(), 'added_rooms']
best_added_rooms_roi = added_rooms_metrics.loc[added_rooms_metrics['avg_roi_rooms'].idxmax(), 'avg_roi_rooms']

st.write(f"""
### Conclusiones Principales:

1. **Rentabilidad por Tipo de Alquiler:**
   - El alquiler por habitaciones ofrece un ROI promedio de **{avg_roi_rooms:.2f}%**, mientras que el alquiler tradicional ofrece **{avg_roi_traditional:.2f}%**.
   - Esto representa una diferencia de **{roi_difference:.2f}%** a favor del alquiler por habitaciones.

2. **Mejores Zonas para Invertir:**
   - El distrito con mayor rentabilidad es **{best_district}** con un ROI de **{best_district_roi:.2f}%**.
   - El barrio más rentable es **{best_neighborhood}** con un ROI de **{best_neighborhood_roi:.2f}%**.

3. **Impacto de Añadir Habitaciones:**
   - El escenario óptimo es añadir **{int(best_added_rooms)}** habitaciones, lo que resulta en un ROI promedio de **{best_added_rooms_roi:.2f}%**.
   - Sin embargo, esto debe evaluarse caso por caso, considerando el tamaño de la propiedad y la viabilidad de la reforma.

4. **Factores Clave para la Rentabilidad:**
   - El precio de compra y el tamaño de la propiedad son factores determinantes.
   - Las propiedades más pequeñas tienden a ofrecer mejor ROI cuando se alquilan por habitaciones.
   - La ubicación es crucial: algunos distritos ofrecen rentabilidades significativamente mayores.
""")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
Esta aplicación analiza la rentabilidad potencial de propiedades en Madrid si se alquilan por habitaciones.
Utiliza los filtros para personalizar el análisis según tus intereses.
""")
