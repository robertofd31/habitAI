import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Configuración de la página
st.set_page_config(
    page_title="Recomendación Personalizada",
    page_icon="🏠",
    layout="wide"
)

# Cargar variables de entorno
load_dotenv()

# Función para calcular la hipoteca
def calculate_mortgage(price, down_payment_percentage, interest_rate, years):
    down_payment = price * (down_payment_percentage / 100)
    loan_amount = price - down_payment
    monthly_interest_rate = interest_rate / 100 / 12
    months = years * 12
    monthly_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** months) / ((1 + monthly_interest_rate) ** months - 1)
    return down_payment, loan_amount, monthly_payment

# Función para calcular la rentabilidad
def calculate_roi(property_data, room_price, occupancy_rate, renovation_cost_per_m2, added_rooms=0):
    # Datos básicos de la propiedad
    price = property_data['price']
    size = property_data['size']
    rooms = property_data['rooms'] + added_rooms
    
    # Costes de adquisición
    purchase_tax = price * 0.08  # Impuesto de transmisiones patrimoniales (8%)
    notary_fees = 600  # Gastos de notaría aproximados
    registry_fees = 400  # Gastos de registro aproximados
    
    # Coste de reforma
    renovation_cost = size * renovation_cost_per_m2
    
    # Inversión total
    total_investment = price + purchase_tax + notary_fees + registry_fees + renovation_cost
    
    # Ingresos por alquiler tradicional (estimación)
    monthly_rent_traditional = size * 12  # Estimación de 12€/m² al mes
    annual_income_traditional = monthly_rent_traditional * 12
    
    # Ingresos por alquiler por habitaciones
    monthly_income_rooms = room_price * rooms * (occupancy_rate / 100)
    annual_income_rooms = monthly_income_rooms * 12
    
    # Gastos anuales (estimación)
    property_tax = price * 0.006  # IBI aproximado
    community_fees = 50 * 12  # Gastos de comunidad mensuales
    maintenance = size * 5  # Mantenimiento anual estimado
    
    # Gastos adicionales para alquiler por habitaciones
    additional_expenses_rooms = annual_income_rooms * 0.1  # 10% adicional para gestión, suministros, etc.
    
    # Gastos totales
    annual_expenses_traditional = property_tax + community_fees + maintenance
    annual_expenses_rooms = property_tax + community_fees + maintenance + additional_expenses_rooms
    
    # Beneficio neto anual
    net_income_traditional = annual_income_traditional - annual_expenses_traditional
    net_income_rooms = annual_income_rooms - annual_expenses_rooms
    
    # ROI anual
    annual_roi_traditional = (net_income_traditional / total_investment) * 100
    annual_roi_rooms = (net_income_rooms / total_investment) * 100
    
    # Tiempo de recuperación de la inversión (en años)
    payback_period_traditional = total_investment / net_income_traditional if net_income_traditional > 0 else float('inf')
    payback_period_rooms = total_investment / net_income_rooms if net_income_rooms > 0 else float('inf')
    
    return {
        'total_investment': total_investment,
        'monthly_rent_traditional': monthly_rent_traditional,
        'annual_income_traditional': annual_income_traditional,
        'monthly_income_rooms': monthly_income_rooms,
        'annual_income_rooms': annual_income_rooms,
        'annual_expenses_traditional': annual_expenses_traditional,
        'annual_expenses_rooms': annual_expenses_rooms,
        'net_income_traditional': net_income_traditional,
        'net_income_rooms': net_income_rooms,
        'annual_roi_traditional': annual_roi_traditional,
        'annual_roi_rooms': annual_roi_rooms,
        'payback_period_traditional': payback_period_traditional,
        'payback_period_rooms': payback_period_rooms
    }

# Función para generar texto de recomendación con OpenAI
def generate_recommendation(property_data, user_data, roi_data):
    try:
        # Configurar el cliente de OpenAI
        client = OpenAI(api_key=st.session_state.openai_api_key)
        
        # Preparar los datos para el prompt
        property_info = {
            "address": property_data['address'] if pd.notna(property_data['address']) else "No disponible",
            "price": f"{int(property_data['price']):,}€" if pd.notna(property_data['price']) else "No disponible",
            "size": f"{int(property_data['size'])}m²" if pd.notna(property_data['size']) else "No disponible",
            "rooms": int(property_data['rooms']) if pd.notna(property_data['rooms']) else "No disponible",
            "district": property_data['district'] if pd.notna(property_data['district']) else "No disponible",
            "neighborhood": property_data['neighborhood'] if pd.notna(property_data['neighborhood']) else "No disponible"
        }
        
        # Datos financieros
        financial_data = {
            "total_investment": f"{int(roi_data['total_investment']):,}€",
            "monthly_payment": f"{int(user_data['monthly_payment']):,}€",
            "roi_traditional": f"{roi_data['annual_roi_traditional']:.2f}%",
            "roi_rooms": f"{roi_data['annual_roi_rooms']:.2f}%",
            "payback_traditional": f"{roi_data['payback_period_traditional']:.1f} años",
            "payback_rooms": f"{roi_data['payback_period_rooms']:.1f} años",
            "renovation_cost": f"{int(user_data['renovation_budget']):,}€"
        }
        
        # Crear el prompt
        prompt = f"""
        Un usuario está buscando una propiedad en la zona de {user_data['zone']} con un presupuesto de {user_data['budget']:,}€.
        El usuario planea financiar el {user_data['financing_percentage']}% del precio con un interés del {user_data['interest_rate']}% anual a {user_data['years']} años.
        
        Basándonos en los datos, recomendamos la siguiente propiedad:
        
        DATOS DE LA PROPIEDAD:
        - Dirección: {property_info['address']}
        - Precio: {property_info['price']}
        - Tamaño: {property_info['size']}
        - Habitaciones: {property_info['rooms']}
        - Distrito: {property_info['district']}
        - Barrio: {property_info['neighborhood']}
        
        DATOS FINANCIEROS:
        - Inversión total (incluyendo impuestos y reforma): {financial_data['total_investment']}
        - Cuota mensual de hipoteca: {financial_data['monthly_payment']}
        - Rentabilidad anual (alquiler tradicional): {financial_data['roi_traditional']}
        - Rentabilidad anual (alquiler por habitaciones): {financial_data['roi_rooms']}
        - Tiempo de recuperación (alquiler tradicional): {financial_data['payback_traditional']}
        - Tiempo de recuperación (alquiler por habitaciones): {financial_data['payback_rooms']}
        - Presupuesto para reforma: {financial_data['renovation_cost']}
        
        Por favor, genera un texto persuasivo y profesional de aproximadamente 250 palabras explicando:
        1. Por qué esta propiedad es una buena opción para el usuario
        2. Las ventajas financieras de la inversión
        3. La comparación entre alquiler tradicional y alquiler por habitaciones
        4. Recomendaciones sobre la reforma
        5. Conclusión con una recomendación clara
        
        Usa un tono profesional pero accesible, y destaca los datos más relevantes para un inversor inmobiliario.
        """
        
        # Llamar a la API de OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asesor inmobiliario experto en inversiones y rentabilidad."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extraer y devolver el texto generado
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"Error al generar la recomendación: {e}")
        return f"""
        No se pudo generar una recomendación personalizada debido a un error: {str(e)}
        
        Sin embargo, basándonos en los datos disponibles, esta propiedad parece ser una opción interesante para inversión, 
        especialmente si se considera el alquiler por habitaciones que ofrece una rentabilidad del {roi_data['annual_roi_rooms']:.2f}%.
        """

# Página principal
def personalized_recommendation_page():
    st.title("Recomendación Personalizada de Propiedades")
    st.markdown("Introduce tus preferencias y te recomendaremos la mejor propiedad para invertir")
    
    # Cargar datos
    try:
        df_properties = pd.read_csv("propiedades_madrid_hasta_200k.csv")
        st.session_state.df_properties = df_properties
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.stop()
    
    # Configuración de la API de OpenAI en el sidebar
    st.sidebar.header("Configuración de API")
    
    # Intentar obtener claves de diferentes fuentes
    try:
        # 1. Intentar obtener de Streamlit Secrets
        openai_key = st.secrets["openai"]
        st.sidebar.success("API key de OpenAI cargada desde Streamlit Secrets")
    except Exception:
        try:
            # 2. Intentar obtener de variables de entorno
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if openai_key:
                st.sidebar.success("API key de OpenAI cargada desde variables de entorno")
            else:
                openai_key = ""
                st.sidebar.warning("No se encontró API key de OpenAI configurada")
        except Exception:
            openai_key = ""
            st.sidebar.warning("No se encontró API key de OpenAI configurada")
    
    # Permitir al usuario introducir o modificar la clave
    with st.sidebar.expander("Configurar API Key", expanded=not openai_key):
        api_key = st.text_input("OpenAI API Key", type="password", value=openai_key)
    
    # Guardar la API key en session_state
    st.session_state.openai_api_key = api_key
    
    # Datos del usuario en el sidebar
    st.sidebar.header("Tus Preferencias")
    budget = st.sidebar.number_input("Presupuesto (€)", min_value=50000, max_value=1000000, step=1000, value=200000)
    
    # Filtrar distritos disponibles según el presupuesto
    available_districts = df_properties[df_properties['price'] <= budget]['district'].dropna().unique()
    if len(available_districts) == 0:
        st.warning(f"No hay propiedades disponibles con un presupuesto de {budget}€. Por favor, aumenta tu presupuesto.")
        return
    
    zone = st.sidebar.selectbox("Zona (Distrito)", options=sorted(available_districts))
    
    # Datos financieros
    st.sidebar.subheader("Datos Financieros")
    financing_percentage = st.sidebar.slider("Porcentaje de Financiación (%)", min_value=0, max_value=100, step=5, value=80)
    interest_rate = st.sidebar.slider("Interés Anual (%)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
    years = st.sidebar.slider("Años de Hipoteca", min_value=5, max_value=30, step=1, value=25)
    
    # Datos de rentabilidad
    st.sidebar.subheader("Datos de Rentabilidad")
    room_price = st.sidebar.number_input("Precio de Alquiler por Habitación (€/mes)", min_value=300, max_value=1000, step=50, value=450)
    occupancy_rate = st.sidebar.slider("Tasa de Ocupación (%)", min_value=50, max_value=100, step=5, value=90)
    renovation_cost_per_m2 = st.sidebar.slider("Coste de Reforma por m² (€)", min_value=0, max_value=1000, step=50, value=300)
    renovation_budget = st.sidebar.number_input("Presupuesto Total para Reforma (€)", min_value=0, max_value=100000, step=1000, value=15000)
    
    # Botón para buscar recomendación
    if st.sidebar.button("Buscar Recomendación"):
        if not api_key:
            st.error("Por favor, introduce tu API Key de OpenAI para obtener una recomendación personalizada.")
            return
        
        # Filtrar propiedades según el presupuesto y la zona
        filtered_properties = df_properties[
            (df_properties['price'] <= budget) &
            (df_properties['district'] == zone)
        ]
        
        if filtered_properties.empty:
            st.warning(f"No se encontraron propiedades en {zone} dentro de tu presupuesto. Intenta ajustar tus criterios.")
            return
        
        # Mostrar spinner mientras se procesa
        with st.spinner("Analizando propiedades y generando recomendación..."):
            # Calcular rentabilidad para cada propiedad
            properties_with_roi = []
            
            for _, property_data in filtered_properties.iterrows():
                # Calcular hipoteca
                down_payment, loan_amount, monthly_payment = calculate_mortgage(
                    property_data['price'],
                    financing_percentage,
                    interest_rate,
                    years
                )
                
                # Calcular ROI
                roi_data = calculate_roi(
                    property_data,
                    room_price,
                    occupancy_rate,
                    renovation_cost_per_m2
                )
                
                properties_with_roi.append({
                    'property_data': property_data,
                    'roi_data': roi_data,
                    'monthly_payment': monthly_payment,
                    'down_payment': down_payment,
                    'loan_amount': loan_amount
                })
            
            # Ordenar propiedades por ROI (alquiler por habitaciones)
            sorted_properties = sorted(
                properties_with_roi,
                key=lambda x: x['roi_data']['annual_roi_rooms'],
                reverse=True
            )
            
            # Seleccionar la propiedad con mejor ROI
            best_property = sorted_properties[0]
            
            # Datos para la recomendación
            user_data = {
                'budget': budget,
                'zone': zone,
                'financing_percentage': financing_percentage,
                'interest_rate': interest_rate,
                'years': years,
                'monthly_payment': best_property['monthly_payment'],
                'renovation_budget': renovation_budget
            }
            
            # Generar recomendación con OpenAI
            recommendation_text = generate_recommendation(
                best_property['property_data'],
                user_data,
                best_property['roi_data']
            )
            
            # Mostrar resultados
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("Propiedad Recomendada")
                property_data = best_property['property_data']
                roi_data = best_property['roi_data']
                
                # Información básica de la propiedad
                st.subheader(f"{property_data['address'] if pd.notna(property_data['address']) else 'Dirección no disponible'}")
                st.write(f"**ID de la propiedad:** {property_data['propertyCode']}")
                st.write(f"**Precio:** {int(property_data['price']):,}€")
                st.write(f"**Tamaño:** {int(property_data['size'])}m²")
                st.write(f"**Habitaciones:** {int(property_data['rooms'])}")
                st.write(f"**Distrito:** {property_data['district']}")
                st.write(f"**Barrio:** {property_data['neighborhood']}")
                
                # Recomendación generada por OpenAI
                st.header("Análisis y Recomendación")
                st.write(recommendation_text)
            
            with col2:
                # Datos financieros
                st.header("Datos Financieros")
                
                # Hipoteca
                st.subheader("Hipoteca")
                st.metric("Entrada ({}%)".format(financing_percentage), f"{int(best_property['down_payment']):,}€")
                st.metric("Préstamo", f"{int(best_property['loan_amount']):,}€")
                st.metric("Cuota Mensual", f"{int(best_property['monthly_payment']):,}€")
                
                # Rentabilidad
                st.subheader("Rentabilidad")
                
                # Crear dos columnas para comparar alquiler tradicional vs por habitaciones
                col_trad, col_rooms = st.columns(2)
                
                with col_trad:
                    st.markdown("**Alquiler Tradicional**")
                    st.metric("ROI Anual", f"{roi_data['annual_roi_traditional']:.2f}%")
                    st.metric("Ingreso Mensual", f"{int(roi_data['monthly_rent_traditional']):,}€")
                    st.metric("Recuperación", f"{roi_data['payback_period_traditional']:.1f} años")
                
                with col_rooms:
                    st.markdown("**Alquiler por Habitaciones**")
                    st.metric("ROI Anual", f"{roi_data['annual_roi_rooms']:.2f}%")
                    st.metric("Ingreso Mensual", f"{int(roi_data['monthly_income_rooms']):,}€")
                    st.metric("Recuperación", f"{roi_data['payback_period_rooms']:.1f} años")
                
                # Reforma
                st.subheader("Reforma")
                st.metric("Presupuesto", f"{renovation_budget:,}€")
                st.metric("Coste por m²", f"{renovation_cost_per_m2}€/m²")
                
                # Botón para ver detalles completos
                if st.button("Ver Detalles Completos"):
                    st.session_state.selected_property_code = property_data['propertyCode']
                    st.experimental_rerun()

# Ejecutar la página
if __name__ == "__main__":
    personalized_recommendation_page()
