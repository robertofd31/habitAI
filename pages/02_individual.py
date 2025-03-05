import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import re
from PIL import Image
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Detalle de Propiedad",
    page_icon="🏠",
    layout="wide"
)

# Título principal
st.title("🏠 Detalle de Propiedad Individual")
st.write("Análisis detallado de rentabilidad y características de una propiedad específica.")

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

# Sidebar para seleccionar propiedad
st.sidebar.header("Selección de Propiedad")

# Filtros para encontrar propiedades
district_options = sorted(df_properties['district'].dropna().unique())
selected_district = st.sidebar.selectbox(
    "Distrito",
    options=district_options
)

# Filtrar barrios por distrito seleccionado
neighborhood_options = sorted(df_properties[df_properties['district'] == selected_district]['neighborhood'].dropna().unique())
selected_neighborhood = st.sidebar.selectbox(
    "Barrio",
    options=neighborhood_options
)

# Filtrar propiedades por barrio seleccionado
filtered_properties = df_properties[
    (df_properties['district'] == selected_district) &
    (df_properties['neighborhood'] == selected_neighborhood)
]

# Crear opciones para el selectbox de propiedades
property_options = []
for _, row in filtered_properties.iterrows():
    address = row['address'] if pd.notna(row['address']) else "Sin dirección"
    price = f"{int(row['price']):,}€" if pd.notna(row['price']) else "Precio desconocido"
    size = f"{int(row['size'])}m²" if pd.notna(row['size']) else "Tamaño desconocido"
    rooms = f"{int(row['rooms'])} hab" if pd.notna(row['rooms']) else "Habitaciones desconocidas"

    option_text = f"{address} - {price} - {size} - {rooms}"
    property_options.append((option_text, row['propertyCode']))

# Crear lista de textos para mostrar en el selectbox
property_texts = [text for text, _ in property_options]
property_codes = [code for _, code in property_options]

# Selectbox para elegir propiedad
selected_property_index = st.sidebar.selectbox(
    "Selecciona una propiedad",
    range(len(property_texts)),
    format_func=lambda i: property_texts[i]
)

selected_property_code = property_codes[selected_property_index]
property_data = df_properties[df_properties['propertyCode'] == selected_property_code].iloc[0]

# Parámetros para el cálculo de rentabilidad
st.sidebar.header("Parámetros de Rentabilidad")

# Precio medio de alquiler por habitación (por distrito)
avg_room_price_by_district = df_rooms.groupby('district')['price'].mean().to_dict()
default_room_price = int(df_rooms['price'].mean())

# Precio medio de habitación en el distrito seleccionado
district_avg_price = avg_room_price_by_district.get(selected_district, default_room_price)

# Permitir ajustar el precio de alquiler por habitación
room_price = st.sidebar.slider(
    f"Precio de alquiler por habitación (€)",
    min_value=int(district_avg_price * 0.5),
    max_value=int(district_avg_price * 1.5),
    value=int(district_avg_price),
    step=10
)

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

# Función para calcular la rentabilidad
def calculate_roi(property_data, added_rooms=0, room_price=room_price):
    # Número total de habitaciones (originales + añadidas)
    total_rooms = property_data['rooms'] + added_rooms

    # Verificar si el tamaño permite añadir habitaciones (mínimo 10m² por habitación)
    if property_data['size'] / total_rooms < 10:
        total_rooms = max(1, int(property_data['size'] / 10))  # Al menos una habitación

    # Ingresos mensuales por alquiler de habitaciones
    monthly_income_rooms = total_rooms * room_price * (occupancy_rate / 100)

    # Ingresos mensuales por alquiler tradicional
    monthly_income_traditional = (property_data['price'] * (traditional_rental_yield / 100)) / 12

    # Gastos mensuales
    monthly_expenses_rooms = monthly_income_rooms * (monthly_expenses_percent / 100)
    monthly_expenses_traditional = monthly_income_traditional * (monthly_expenses_percent / 100)

    # Beneficio neto mensual
    net_monthly_profit_rooms = monthly_income_rooms - monthly_expenses_rooms
    net_monthly_profit_traditional = monthly_income_traditional - monthly_expenses_traditional

    # Inversión inicial
    initial_investment = property_data['price']

    # Coste de reforma
    renovation_cost = property_data['size'] * renovation_cost_per_m2

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
        'monthly_expenses_rooms': monthly_expenses_rooms,
        'monthly_expenses_traditional': monthly_expenses_traditional,
        'net_monthly_profit_rooms': net_monthly_profit_rooms,
        'net_monthly_profit_traditional': net_monthly_profit_traditional,
        'renovation_cost': renovation_cost,
        'room_addition_cost': room_addition_cost,
        'total_investment': total_investment,
        'annual_roi_rooms': annual_roi_rooms,
        'annual_roi_traditional': annual_roi_traditional,
        'payback_period_rooms': payback_period_rooms,
        'payback_period_traditional': payback_period_traditional
    }

# Calcular rentabilidad para diferentes escenarios
base_roi = calculate_roi(property_data)
roi_1_room = calculate_roi(property_data, added_rooms=1)
roi_2_rooms = calculate_roi(property_data, added_rooms=2)

# Mostrar información de la propiedad
st.header("Información de la Propiedad")

# Mostrar imagen de la propiedad si está disponible
if pd.notna(property_data['thumbnail']):
    try:
        response = requests.get(property_data['thumbnail'])
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Imagen de la propiedad", use_column_width=True)
    except:
        st.warning("No se pudo cargar la imagen de la propiedad.")

# Crear dos columnas para mostrar la información
col1, col2 = st.columns(2)

with col1:
    st.subheader("Características Básicas")
    st.write(f"**Dirección:** {property_data['address'] if pd.notna(property_data['address']) else 'No disponible'}")
    st.write(f"**Distrito:** {property_data['district'] if pd.notna(property_data['district']) else 'No disponible'}")
    st.write(f"**Barrio:** {property_data['neighborhood'] if pd.notna(property_data['neighborhood']) else 'No disponible'}")
    st.write(f"**Precio:** {int(property_data['price']):,} €")
    st.write(f"**Tamaño:** {int(property_data['size'])} m²")
    st.write(f"**Habitaciones:** {int(property_data['rooms'])}")
    st.write(f"**Baños:** {int(property_data['bathrooms']) if pd.notna(property_data['bathrooms']) else 'No disponible'}")
    st.write(f"**Precio por m²:** {int(property_data['price_per_m2']):,} €/m²")

    # Mostrar si tiene ascensor
    has_lift = "Sí" if property_data.get('hasLift') == True else "No"
    st.write(f"**Ascensor:** {has_lift}")

    # Mostrar planta
    floor = property_data.get('floor', 'No disponible')
    st.write(f"**Planta:** {floor}")

with col2:
    st.subheader("Ubicación")
    # Mostrar mapa si hay coordenadas disponibles
    if pd.notna(property_data['latitude']) and pd.notna(property_data['longitude']):
        map_data = pd.DataFrame({
            'lat': [property_data['latitude']],
            'lon': [property_data['longitude']]
        })
        st.map(map_data)
    else:
        st.write("Coordenadas no disponibles para mostrar el mapa.")

    # Mostrar descripción si está disponible
    if pd.notna(property_data['description']):
        st.subheader("Descripción")
        # Limitar la descripción a un número máximo de caracteres
        max_chars = 500
        description = property_data['description']
        if len(description) > max_chars:
            description = description[:max_chars] + "..."
        st.write(description)

        # Botón para mostrar la descripción completa
        if len(property_data['description']) > max_chars:
            if st.button("Mostrar descripción completa"):
                st.write(property_data['description'])

# Análisis de Rentabilidad
st.header("Análisis de Rentabilidad")

# Crear pestañas para diferentes escenarios
tab1, tab2, tab3 = st.tabs(["Configuración Actual", "Añadir 1 Habitación", "Añadir 2 Habitaciones"])

# Función para mostrar métricas de rentabilidad
def show_roi_metrics(roi_data, added_rooms=0):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ROI Anual (Alquiler por Habitaciones)",
            f"{roi_data['annual_roi_rooms']:.2f}%"
        )
        st.metric(
            "Beneficio Neto Mensual",
            f"{roi_data['net_monthly_profit_rooms']:.2f} €"
        )
        st.metric(
            "Tiempo de Recuperación",
            f"{roi_data['payback_period_rooms']:.2f} años" if roi_data['payback_period_rooms'] != float('inf') else "∞"
        )

    with col2:
        st.metric(
            "ROI Anual (Alquiler Tradicional)",
            f"{roi_data['annual_roi_traditional']:.2f}%"
        )
        st.metric(
            "Beneficio Neto Mensual (Tradicional)",
            f"{roi_data['net_monthly_profit_traditional']:.2f} €"
        )
        st.metric(
            "Tiempo de Recuperación (Tradicional)",
            f"{roi_data['payback_period_traditional']:.2f} años" if roi_data['payback_period_traditional'] != float('inf') else "∞"
        )

    with col3:
        st.metric(
            "Diferencia de ROI",
            f"{roi_data['annual_roi_rooms'] - roi_data['annual_roi_traditional']:.2f}%",
            delta=f"{roi_data['annual_roi_rooms'] - roi_data['annual_roi_traditional']:.2f}%"
        )
        st.metric(
            "Habitaciones Totales",
            f"{roi_data['total_rooms']}"
        )
        st.metric(
            "Inversión Total",
            f"{roi_data['total_investment']:,.2f} €"
        )

    # Mostrar desglose de ingresos y gastos
    st.subheader("Desglose Financiero Mensual")

    # Crear datos para el gráfico
    categories = ['Ingresos', 'Gastos', 'Beneficio Neto']

    # Datos para alquiler por habitaciones
    values_rooms = [
        roi_data['monthly_income_rooms'],
        roi_data['monthly_expenses_rooms'],
        roi_data['net_monthly_profit_rooms']
    ]

    # Datos para alquiler tradicional
    values_traditional = [
        roi_data['monthly_income_traditional'],
        roi_data['monthly_expenses_traditional'],
        roi_data['net_monthly_profit_traditional']
    ]

    # Crear gráfico de barras
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=values_rooms,
        name='Alquiler por Habitaciones',
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=values_traditional,
        name='Alquiler Tradicional',
        marker_color='#ff7f0e'
    ))

    fig.update_layout(
        title='Comparativa Financiera Mensual',
        xaxis_title='Categoría',
        yaxis_title='Euros (€)',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Mostrar desglose de la inversión
    st.subheader("Desglose de la Inversión")

    # Datos para el gráfico de pastel
    investment_labels = ['Precio de Compra', 'Coste de Reforma']
    investment_values = [property_data['price'], roi_data['renovation_cost']]

    # Añadir coste de habitaciones adicionales si es aplicable
    if added_rooms > 0:
        investment_labels.append('Coste de Habitaciones Adicionales')
        investment_values.append(roi_data['room_addition_cost'])

    # Crear gráfico de pastel
    fig = px.pie(
        values=investment_values,
        names=investment_labels,
        title='Distribución de la Inversión',
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')

    st.plotly_chart(fig, use_container_width=True)

# Mostrar métricas para cada escenario
with tab1:
    st.subheader("Rentabilidad con Configuración Actual")
    show_roi_metrics(base_roi)

with tab2:
    st.subheader("Rentabilidad Añadiendo 1 Habitación")
    show_roi_metrics(roi_1_room, added_rooms=1)

with tab3:
    st.subheader("Rentabilidad Añadiendo 2 Habitaciones")
    show_roi_metrics(roi_2_rooms, added_rooms=2)

# Comparativa de escenarios
st.header("Comparativa de Escenarios")

# Crear datos para el gráfico
scenarios = ['Configuración Actual', 'Añadir 1 Habitación', 'Añadir 2 Habitaciones']
roi_values = [base_roi['annual_roi_rooms'], roi_1_room['annual_roi_rooms'], roi_2_rooms['annual_roi_rooms']]
payback_values = [base_roi['payback_period_rooms'], roi_1_room['payback_period_rooms'], roi_2_rooms['payback_period_rooms']]

# Crear gráfico de barras con dos ejes Y
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(
        x=scenarios,
        y=roi_values,
        name='ROI Anual (%)',
        marker_color='#1f77b4'
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=scenarios,
        y=payback_values,
        name='Tiempo de Recuperación (años)',
        marker_color='#ff7f0e',
        mode='lines+markers'
    ),
    secondary_y=True
)

fig.update_layout(
    title='Comparativa de ROI y Tiempo de Recuperación por Escenario',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.update_yaxes(title_text='ROI Anual (%)', secondary_y=False)
fig.update_yaxes(title_text='Tiempo de Recuperación (años)', secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Análisis de sensibilidad
st.header("Análisis de Sensibilidad")

# Seleccionar variable para análisis de sensibilidad
sensitivity_variable = st.selectbox(
    "Selecciona variable para análisis de sensibilidad",
    ["Precio de alquiler por habitación", "Tasa de ocupación", "Coste de reforma por m²"]
)

def calculate_sensitivity(variable, property_data, added_rooms=0):
    # Declarar las variables globales al inicio de la función
    global occupancy_rate, renovation_cost_per_m2

    results = []

    if variable == "Precio de alquiler por habitación":
        # Variar el precio de alquiler por habitación
        min_price = int(room_price * 0.5)
        max_price = int(room_price * 1.5)
        step = int((max_price - min_price) / 10)

        for price in range(min_price, max_price + step, step):
            roi = calculate_roi(property_data, added_rooms=added_rooms, room_price=price)
            results.append({
                'variable_value': price,
                'roi': roi['annual_roi_rooms'],
                'payback': roi['payback_period_rooms']
            })

        x_label = "Precio de Alquiler por Habitación (€)"

    elif variable == "Tasa de ocupación":
        # Variar la tasa de ocupación
        original_rate = occupancy_rate  # Guardar valor original

        for occ_rate in range(50, 101, 5):
            # Modificar temporalmente la variable global
            occupancy_rate = occ_rate

            roi = calculate_roi(property_data, added_rooms=added_rooms)
            results.append({
                'variable_value': occ_rate,
                'roi': roi['annual_roi_rooms'],
                'payback': roi['payback_period_rooms']
            })

        # Restaurar valor original
        occupancy_rate = original_rate
        x_label = "Tasa de Ocupación (%)"

    elif variable == "Coste de reforma por m²":
        # Variar el coste de reforma
        original_cost = renovation_cost_per_m2  # Guardar valor original

        for cost in range(0, 601, 50):
            # Modificar temporalmente la variable global
            renovation_cost_per_m2 = cost

            roi = calculate_roi(property_data, added_rooms=added_rooms)
            results.append({
                'variable_value': cost,
                'roi': roi['annual_roi_rooms'],
                'payback': roi['payback_period_rooms']
            })

        # Restaurar valor original
        renovation_cost_per_m2 = original_cost
        x_label = "Coste de Reforma por m² (€)"

    return results, x_label

# Calcular sensibilidad para el escenario seleccionado
sensitivity_results, x_label = calculate_sensitivity(sensitivity_variable, property_data, added_rooms=0)

# Crear DataFrame para el gráfico
sensitivity_df = pd.DataFrame(sensitivity_results)

# Crear gráfico de sensibilidad
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=sensitivity_df['variable_value'],
        y=sensitivity_df['roi'],
        name='ROI Anual (%)',
        mode='lines+markers',
        marker_color='#1f77b4'
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=sensitivity_df['variable_value'],
        y=sensitivity_df['payback'],
        name='Tiempo de Recuperación (años)',
        mode='lines+markers',
        marker_color='#ff7f0e'
    ),
    secondary_y=True
)

fig.update_layout(
    title=f'Análisis de Sensibilidad: Impacto de {sensitivity_variable} en la Rentabilidad',
    xaxis_title=x_label,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.update_yaxes(title_text='ROI Anual (%)', secondary_y=False)
fig.update_yaxes(title_text='Tiempo de Recuperación (años)', secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Conclusiones y recomendaciones
st.header("Conclusiones y Recomendaciones")

# Determinar el mejor escenario
roi_scenarios = [
    {"name": "Configuración Actual", "roi": base_roi['annual_roi_rooms'], "data": base_roi},
    {"name": "Añadir 1 Habitación", "roi": roi_1_room['annual_roi_rooms'], "data": roi_1_room},
    {"name": "Añadir 2 Habitaciones", "roi": roi_2_rooms['annual_roi_rooms'], "data": roi_2_rooms}
]

best_scenario = max(roi_scenarios, key=lambda x: x['roi'])

# Comparar con alquiler tradicional
roi_difference = best_scenario['roi'] - best_scenario['data']['annual_roi_traditional']

# Generar conclusiones
st.write(f"""
### Conclusiones para esta propiedad:

1. **Mejor escenario:** {best_scenario['name']} con un ROI anual de **{best_scenario['roi']:.2f}%**.

2. **Comparación con alquiler tradicional:** El alquiler por habitaciones ofrece un ROI **{roi_difference:.2f}%**
   {'superior' if roi_difference > 0 else 'inferior'} al alquiler tradicional en el mejor escenario.

3. **Tiempo de recuperación:** La inversión se recuperaría en aproximadamente **{best_scenario['data']['payback_period_rooms']:.2f} años**
   con el alquiler por habitaciones en el mejor escenario.

4. **Inversión total requerida:** **{best_scenario['data']['total_investment']:,.2f} €**, que incluye el precio de compra,
   costes de reforma y {0 if best_scenario['name'] == 'Configuración Actual' else '1' if best_scenario['name'] == 'Añadir 1 Habitación' else '2'}
   habitaciones adicionales.

5. **Beneficio neto mensual:** **{best_scenario['data']['net_monthly_profit_rooms']:.2f} €** con el alquiler por habitaciones
   en el mejor escenario.
""")

# Recomendaciones basadas en el análisis
st.write("### Recomendaciones:")

if best_scenario['roi'] < 4:
    st.write("""
    - **Rentabilidad baja:** Esta propiedad ofrece una rentabilidad por debajo del 4%, lo que podría considerarse bajo
      para una inversión inmobiliaria. Considera buscar otras opciones o negociar un precio de compra más bajo.
    """)
elif best_scenario['roi'] < 6:
    st.write("""
    - **Rentabilidad moderada:** Esta propiedad ofrece una rentabilidad aceptable, pero no excepcional.
      Podría ser una buena inversión a largo plazo, especialmente si esperas que la zona se revalorice.
    """)
else:
    st.write("""
    - **Rentabilidad alta:** Esta propiedad ofrece una excelente rentabilidad. Considera avanzar con la inversión
      siguiendo el escenario recomendado.
    """)

if best_scenario['name'] != "Configuración Actual":
    st.write(f"""
    - **Reforma recomendada:** Implementar {best_scenario['name'].lower()} mejoraría significativamente la rentabilidad.
      Asegúrate de verificar que la distribución del espacio lo permite y que cumple con la normativa local.
    """)

if sensitivity_variable == "Precio de alquiler por habitación":
    max_roi_price = sensitivity_df.loc[sensitivity_df['roi'].idxmax(), 'variable_value']
    st.write(f"""
    - **Precio óptimo por habitación:** Según el análisis de sensibilidad, el precio óptimo por habitación sería
      aproximadamente **{max_roi_price} €** para maximizar el ROI.
    """)

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
Esta página muestra un análisis detallado de una propiedad específica, incluyendo sus características,
ubicación, rentabilidad potencial y análisis de sensibilidad.
""")
