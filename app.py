import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Habitaciones Madrid",
    page_icon="🏠",
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
    return df

# Función para mostrar una habitación
def display_room(room):
    col1, col2 = st.columns([1, 2])

    with col1:
        if pd.notna(room['thumbnail']):
            st.image(room['thumbnail'], width=200)
        else:
            st.image("https://via.placeholder.com/200x150?text=No+Image", width=200)

    with col2:
        st.subheader(f"{room['propertyType']} en {room['address']}")
        st.write(f"**Precio:** {room['price']}€/mes")
        st.write(f"**Tamaño:** {room['size']} m² | **Habitaciones:** {room['rooms']} | **Baños:** {room['bathrooms']}")
        st.write(f"**Barrio:** {room['neighborhood']}, {room['district']}")

        # Mostrar descripción con botón de expansión
        with st.expander("Ver descripción"):
            st.write(room['description'])

        # Enlace a la propiedad
        if pd.notna(room['url']):
            st.markdown(f"[Ver en Idealista]({room['url']})")

# Título principal
st.title("🏠 Habitaciones en Madrid")

# Cargar datos
df = load_data()

# Sidebar para filtros
st.sidebar.header("Filtros")

# Filtro de precio
min_price = int(df['price'].min())
max_price = int(df['price'].max())
price_range = st.sidebar.slider(
    "Rango de precio (€/mes)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# Filtro de tamaño
min_size = int(df['size'].min())
max_size = int(df['size'].max())
size_range = st.sidebar.slider(
    "Tamaño (m²)",
    min_value=min_size,
    max_value=max_size,
    value=(min_size, max_size)
)

# Filtro de habitaciones
rooms_options = sorted(df['rooms'].dropna().unique())
selected_rooms = st.sidebar.multiselect(
    "Número de habitaciones",
    options=rooms_options,
    default=rooms_options
)

# Filtro de baños
bathrooms_options = sorted(df['bathrooms'].dropna().unique())
selected_bathrooms = st.sidebar.multiselect(
    "Número de baños",
    options=bathrooms_options,
    default=bathrooms_options
)

# Filtro de distritos
district_options = sorted(df['district'].dropna().unique())
selected_districts = st.sidebar.multiselect(
    "Distrito",
    options=district_options,
    default=[]
)

# Filtro de barrios
neighborhood_options = sorted(df['neighborhood'].dropna().unique())
selected_neighborhoods = st.sidebar.multiselect(
    "Barrio",
    options=neighborhood_options,
    default=[]
)

# Filtro por ascensor
has_lift_options = ["Todos", "Con ascensor", "Sin ascensor"]
has_lift_filter = st.sidebar.radio("Ascensor", has_lift_options)

# Aplicar filtros
filtered_df = df.copy()

# Filtro de precio
filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) &
                          (filtered_df['price'] <= price_range[1])]

# Filtro de tamaño
filtered_df = filtered_df[(filtered_df['size'] >= size_range[0]) &
                          (filtered_df['size'] <= size_range[1])]

# Filtro de habitaciones
if selected_rooms:
    filtered_df = filtered_df[filtered_df['rooms'].isin(selected_rooms)]

# Filtro de baños
if selected_bathrooms:
    filtered_df = filtered_df[filtered_df['bathrooms'].isin(selected_bathrooms)]

# Filtro de distritos
if selected_districts:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]

# Filtro de barrios
if selected_neighborhoods:
    filtered_df = filtered_df[filtered_df['neighborhood'].isin(selected_neighborhoods)]

# Filtro de ascensor
if has_lift_filter == "Con ascensor":
    filtered_df = filtered_df[filtered_df['hasLift'] == True]
elif has_lift_filter == "Sin ascensor":
    filtered_df = filtered_df[filtered_df['hasLift'] == False]

# Mostrar número de resultados
st.write(f"**{len(filtered_df)} habitaciones encontradas**")

# Paginación
items_per_page = 10
total_pages = (len(filtered_df) + items_per_page - 1) // items_per_page

if total_pages > 0:
    page_number = st.number_input(
        "Página",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

    # Calcular índices para la página actual
    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_df))

    # Mostrar paginación
    st.write(f"Mostrando {start_idx + 1}-{end_idx} de {len(filtered_df)} habitaciones")

    # Mostrar habitaciones para la página actual
    current_page_df = filtered_df.iloc[start_idx:end_idx]

    # Mostrar cada habitación
    for _, room in current_page_df.iterrows():
        display_room(room)
        st.markdown("---")

    # Botones de navegación
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page_number > 1:
            if st.button("← Anterior"):
                page_number -= 1

    with col3:
        if page_number < total_pages:
            if st.button("Siguiente →"):
                page_number += 1
else:
    st.write("No se encontraron habitaciones con los filtros seleccionados.")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
Esta aplicación muestra habitaciones disponibles en Madrid.
Utiliza los filtros para encontrar la habitación que mejor se adapte a tus necesidades.
""")
