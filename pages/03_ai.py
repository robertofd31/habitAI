import streamlit as st
import pandas as pd
import numpy as np
import openai

# Configuración de la API de OpenAI
openai.api_key = api_key=st.secrets["openai"]  # Reemplaza con tu clave de API

# Función para calcular la hipoteca
def calculate_mortgage(price, down_payment_percentage, interest_rate, years):
    down_payment = price * (down_payment_percentage / 100)
    loan_amount = price - down_payment
    monthly_interest_rate = interest_rate / 100 / 12
    months = years * 12
    monthly_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** months) / ((1 + monthly_interest_rate) ** months - 1)
    return down_payment, loan_amount, monthly_payment

# Función para generar texto de recomendación con OpenAI
def generate_recommendation(property_data, user_data):
    prompt = f"""
    Un usuario está buscando una propiedad en la zona de {user_data['zone']} con un presupuesto de {user_data['budget']}€.
    El usuario planea financiar el {user_data['financing_percentage']}% del precio con un interés del {user_data['interest_rate']}% anual.
    Basándonos en los datos, recomendamos la siguiente propiedad:

    Dirección: {property_data['address']}
    Precio: {property_data['price']}€
    Tamaño: {property_data['size']}m²
    Habitaciones: {property_data['rooms']}
    Distrito: {property_data['district']}
    Barrio: {property_data['neighborhood']}

    Por favor, genera un texto persuasivo y profesional explicando por qué esta propiedad es una buena opción para el usuario, destacando su rentabilidad y características clave.
    """
    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Página de recomendación personalizada
def personalized_recommendation_page(df_properties):
    st.title("Recomendación Personalizada de Propiedades")

    st.sidebar.header("Datos del Usuario")
    budget = st.sidebar.number_input("Presupuesto (€)", min_value=50000, max_value=1000000, step=1000, value=200000)
    zone = st.sidebar.selectbox("Zona (Distrito)", options=sorted(df_properties['district'].dropna().unique()))
    financing_percentage = st.sidebar.slider("Porcentaje de Financiación (%)", min_value=0, max_value=100, step=5, value=80)
    interest_rate = st.sidebar.slider("Interés Anual (%)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
    years = st.sidebar.slider("Años de Hipoteca", min_value=5, max_value=30, step=1, value=25)
    renovation_budget = st.sidebar.number_input("Presupuesto para Reforma (€)", min_value=0, max_value=100000, step=1000, value=10000)

    # Filtrar propiedades según el presupuesto y la zona
    filtered_properties = df_properties[
        (df_properties['price'] <= budget) &
        (df_properties['district'] == zone)
    ]

    if filtered_properties.empty:
        st.warning("No se encontraron propiedades que coincidan con tus criterios. Intenta ajustar el presupuesto o la zona.")
        return

    # Seleccionar la propiedad recomendada (la más barata dentro del presupuesto)
    recommended_property = filtered_properties.sort_values(by="price").iloc[0]

    # Calcular datos de la hipoteca
    down_payment, loan_amount, monthly_payment = calculate_mortgage(
        recommended_property['price'],
        financing_percentage,
        interest_rate,
        years
    )

    # Generar texto de recomendación con OpenAI
    user_data = {
        "budget": budget,
        "zone": zone,
        "financing_percentage": financing_percentage,
        "interest_rate": interest_rate
    }
    recommendation_text = generate_recommendation(recommended_property, user_data)

    # Mostrar resultados
    st.header("Propiedad Recomendada")
    st.subheader(f"{recommended_property['address']} - {recommended_property['price']}€")
    st.write(f"**Tamaño:** {recommended_property['size']}m²")
    st.write(f"**Habitaciones:** {recommended_property['rooms']}")
    st.write(f"**Distrito:** {recommended_property['district']}")
    st.write(f"**Barrio:** {recommended_property['neighborhood']}")

    st.header("Datos de la Hipoteca")
    st.write(f"**Pago Inicial (Entrada):** {down_payment:,.2f}€")
    st.write(f"**Monto del Préstamo:** {loan_amount:,.2f}€")
    st.write(f"**Cuota Mensual:** {monthly_payment:,.2f}€")

    st.header("Presupuesto de Reforma")
    st.write(f"**Presupuesto para Reforma:** {renovation_budget:,.2f}€")

    st.header("Texto de Recomendación")
    st.write(recommendation_text)

# Cargar datos y ejecutar la página
if __name__ == "__main__":
    # Cargar el dataset de propiedades
    df_properties = pd.read_csv("propiedades_madrid_hasta_200k.csv")  # Asegúrate de usar el archivo correcto
    personalized_recommendation_page(df_properties)
