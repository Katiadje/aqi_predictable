import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
from dotenv import load_dotenv
import requests
load_dotenv()
AQICN_API_KEY = os.getenv('AQICN_API_KEY', 'demo')

def get_live_data(city="paris"):
    """Récupère les données AQI + météo en temps réel"""
    aqicn_key = os.getenv("AQICN_API_KEY", "demo")
    weather_key = os.getenv("OPENWEATHER_API_KEY")

    data = {
        "aqi": None,
        "pm25": 0,
        "pm10": 0,
        "temp": 0,
        "humidity": 0,
        "wind": 0,
        "desc": "N/A"
    }

    # --- AQI (WAQI) ---
    try:
        aqi_url = f"https://api.waqi.info/feed/{city}/?token={aqicn_key}"
        resp = requests.get(aqi_url, timeout=10)
        resp.raise_for_status()  # Raise error si status != 200
        resp_json = resp.json()
        
        if resp_json.get("status") == "ok":
            aqi_data = resp_json["data"]
            data["aqi"] = aqi_data.get("aqi")
            iaqi = aqi_data.get("iaqi", {})
            data["pm25"] = iaqi.get("pm25", {}).get("v", 0)
            data["pm10"] = iaqi.get("pm10", {}).get("v", 0)
        else:
            print(f"❌ AQICN API error: {resp_json.get('data', 'Unknown error')}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur AQICN (réseau): {e}")
    except Exception as e:
        print(f"❌ Erreur AQICN (autre): {e}")

    # --- Météo (OpenWeather) ---
    if weather_key:  # Seulement si la clé existe
        try:
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={weather_key}"
            w_resp = requests.get(weather_url, timeout=10)
            w_resp.raise_for_status()
            w_json = w_resp.json()
            
            data["temp"] = w_json["main"]["temp"]
            data["humidity"] = w_json["main"]["humidity"]
            data["wind"] = w_json["wind"]["speed"]
            data["desc"] = w_json["weather"][0]["description"].capitalize()
            
        except requests.exceptions.RequestException as e:
            print(f"🌧️ Erreur météo (réseau): {e}")
        except Exception as e:
            print(f"🌧️ Erreur météo (autre): {e}")
    else:
        print("⚠️ Clé OpenWeather manquante, météo non disponible")

    return data

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.aqi_utils import AQIUtils, APIHelper, CacheManager, DataValidator
from app.utils.plotting import AQIPlotter, DashboardComponents, MapVisualizations

# Import conditionnel du pipeline (peut ne pas être disponible en prod)
try:
    from pipelines.inference_pipeline import AQIInferencePipeline
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    print("⚠️ Pipeline d'inférence non disponible")

# Configuration de la page
st.set_page_config(
    page_title="AQI Predictor - Prédiction Qualité de l'Air",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class AQIStreamlitApp:
    """Application Streamlit principale pour la prédiction AQI"""
    
    def __init__(self):
        self.inference_pipeline = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialise les variables de session"""
        if 'selected_city' not in st.session_state:
            st.session_state.selected_city = 'barcelona'
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'predictions_cache' not in st.session_state:
            st.session_state.predictions_cache = {}
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
    
    def create_header(self):
        """Crée l'en-tête de l'application"""
        st.markdown("""
        <style>
            .header-container {
                background: linear-gradient(90deg, #00416A 0%, #E4E5E6 100%);
                padding: 2.5rem;
                border-radius: 18px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                margin-bottom: 2.5rem;
                color: white;
                position: relative;
            }
            .header-container h1 {
                font-size: 2.3rem;
                font-weight: 700;
                margin-bottom: 0.3rem;
            }
            .header-container p {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            .header-logo {
                position: absolute;
                top: 1.2rem;
                right: 2rem;
                font-size: 2.5rem;
            }
        </style>
        <div class="header-container">
            <div class="header-logo">🌍</div>
            <h1>AQI Predictor Dashboard</h1>
            <p>Prédiction de la qualité de l'air en temps réel avec Intelligence Artificielle</p>
        </div>
        """, unsafe_allow_html=True)
        
    def create_sidebar(self):
        """Crée la barre latérale avec les contrôles"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Sélection de la ville
            cities = APIHelper.get_available_cities()
            selected_city = st.selectbox(
                "🌍 Choisir une ville",
                cities,
                index=cities.index(st.session_state.selected_city) if st.session_state.selected_city in cities else 0
            )
            st.session_state.selected_city = selected_city
            
            st.divider()
            
            # Options d'affichage
            st.subheader("📊 Options d'affichage")
            show_predictions = st.checkbox("Afficher prédictions", value=True)
            show_pollutants = st.checkbox("Détail polluants", value=True)
            show_trends = st.checkbox("Analyse tendances", value=False)
            show_map = st.checkbox("Carte comparative", value=False)
            
            st.divider()
            
            # Rafraîchissement automatique
            st.subheader("🔄 Mise à jour")
            auto_refresh = st.checkbox("Rafraîchissement auto (5min)", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh
            
            if st.button("🔄 Actualiser maintenant"):
                CacheManager.clear_cache()
                st.rerun()
            
            # Informations sur la dernière mise à jour
            if st.session_state.last_update:
                st.caption(f"Dernière mise à jour: {st.session_state.last_update.strftime('%H:%M:%S')}")
            
            st.divider()
            
            # Informations système
            st.subheader("ℹ️ Informations")
            st.caption("**Source de données:** AQICN.org")
            st.caption("**Modèle:** XGBoost/RandomForest")
            st.caption("**Mise à jour:** Temps réel")
            
            # Légende AQI
            st.subheader("🎨 Échelle AQI")
            aqi_legend = {
                "Bon (0-50)": "#00E400",
                "Modéré (51-100)": "#FFFF00",
                "Malsain sensibles (101-150)": "#FF7E00",
                "Malsain (151-200)": "#FF0000",
                "Très malsain (201-300)": "#8F3F97",
                "Dangereux (300+)": "#7E0023"
            }
            
            for label, color in aqi_legend.items():
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 0.2rem 0;">
                    <div style="width: 15px; height: 15px; background-color: {color}; 
                                border-radius: 3px; margin-right: 0.5rem;"></div>
                    <small>{label}</small>
                </div>
                """, unsafe_allow_html=True)
        
        return {
            'selected_city': selected_city,
            'show_predictions': show_predictions,
            'show_pollutants': show_pollutants,
            'show_trends': show_trends,
            'show_map': show_map
        }
    
    @st.cache_data(ttl=300)  # Cache 5 minutes
    def load_current_data(_self, city: str):
        """Charge les données actuelles pour une ville"""
        api_key = os.getenv('AQICN_API_KEY', 'demo')
        return APIHelper.fetch_city_aqi(city, api_key)
    
    @st.cache_data(ttl=1800)  # Cache 30 minutes
    def load_predictions(_self, city: str):
        """Charge les prédictions pour une ville"""
        try:
            # Simulation des prédictions si le pipeline complet n'est pas disponible
            predictions = []
            base_aqi = np.random.uniform(50, 150)
            
            for day in range(1, 4):
                # Simulation avec variations réalistes
                daily_variation = np.random.normal(0, 15)
                day_aqi = max(10, min(300, base_aqi + daily_variation))
                
                predictions.append({
                    'date': (datetime.now() + timedelta(days=day)).date(),
                    'day': day,
                    'aqi_avg': round(day_aqi, 1),
                    'aqi_min': round(max(10, day_aqi - 10), 1),
                    'aqi_max': round(min(300, day_aqi + 15), 1),
                    'category': AQIUtils.get_aqi_category(day_aqi),
                    'city': city,
                    'hourly_predictions': [
                        {
                            'timestamp': datetime.now() + timedelta(days=day, hours=hour),
                            'hour': hour,
                            'aqi': round(max(10, day_aqi + np.random.normal(0, 5)), 1)
                        } for hour in [6, 12, 18, 24]
                    ]
                })
            
            return predictions
            
        except Exception as e:
            st.error(f"Erreur lors du chargement des prédictions: {e}")
            return []
    
    
    def display_current_status(self, current_data: dict, city: str):
        """Affiche le statut AQI actuel"""
        st.subheader(f"📊 Statut Actuel - {city.capitalize()}")

        # Récupération live (AQI + Météo)
        live = get_live_data(city)

        if not live or not live.get("aqi"):
            st.error("❌ Impossible de récupérer les données temps réel.")
            return

        aqi = live["aqi"]
        # Bande colorée dynamique selon l'AQI
        color = AQIUtils.get_aqi_color(aqi)
        category = AQIUtils.get_aqi_category(aqi)
        st.markdown(f"""
            <div style="
                background:{color}22;
                border-left:8px solid {color};
                padding:1rem 1.5rem;
                border-radius:12px;
                margin-bottom:1.5rem;
            ">
            <h3 style="color:{color};margin:0;">Qualité de l'air : {category}</h3>
            <p style="color:#333;margin-top:0.3rem;">Indice AQI actuel : <strong>{aqi}</strong></p>
            </div>
        """, unsafe_allow_html=True)

        pm25 = live["pm25"]
        pm10 = live["pm10"]
        temp = live["temp"]
        humidity = live["humidity"]
        wind = live["wind"]
        desc = live["desc"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            DashboardComponents.aqi_metric_card(aqi, "AQI Actuel")
        with col2:
            st.metric("PM2.5", f"{pm25} μg/m³")
        with col3:
            st.metric("PM10", f"{pm10} μg/m³")
        with col4:
            st.metric("🌡️ Température", f"{temp:.1f} °C")

        # Section météo complémentaire
        st.markdown(f"""
        **Conditions météo actuelles à {city.capitalize()}**
        - 💧 Humidité : {humidity}%
        - 💨 Vent : {wind} km/h
        - ☁️ Ciel : {desc}
        """)

        # Gauge AQI
        col1, col2 = st.columns([2, 1])
        with col1:
            gauge_fig = AQIPlotter.create_aqi_gauge(aqi)
            st.plotly_chart(gauge_fig, use_container_width=True)
        with col2:
            health_info = AQIUtils.get_health_info(aqi)
            st.markdown(f"""
            ### {health_info['icon']} Recommandations
            **{health_info['message']}**

            {health_info['advice']}
            """)

    
    def display_predictions(self, predictions: list, city: str):
        """Affiche les prédictions AQI"""
        st.subheader(f"🔮 Prédictions 3 jours - {city.capitalize()}")
        
        if not predictions:
            st.warning("⚠️ Aucune prédiction disponible")
            return
        
        # Graphique temporel des prédictions
        timeline_fig = AQIPlotter.create_prediction_timeline(predictions, city)
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Résumé des prédictions par jour
        st.subheader("📅 Résumé par jour")
        
        cols = st.columns(len(predictions))
        
        for i, pred in enumerate(predictions):
            with cols[i]:
                date_str = pred['date'].strftime('%d/%m')
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Jour {pred['day']} - {date_str}</h4>
                    <h2 style="color: {AQIUtils.get_aqi_color(pred['aqi_avg'])}">{pred['aqi_avg']}</h2>
                    <p><strong>{pred['category']}</strong></p>
                    <p><small>Plage: {pred['aqi_min']} - {pred['aqi_max']}</small></p>
                </div>
                """, unsafe_allow_html=True)
    
        # Prédictions horaires détaillées
        hourly_fig = AQIPlotter.create_hourly_forecast(predictions)
        st.plotly_chart(hourly_fig, use_container_width=True)
    
    def display_pollutant_details(self, current_data: dict):
        """Affiche les détails des polluants"""
        st.subheader("🏭 Détail des Polluants")
        
        if not current_data:
            st.warning("Données de polluants non disponibles")
            return
        
        iaqi = current_data.get('iaqi', {})
        
        # Graphique en barres des polluants
        pollutant_data = {
            'pm25': iaqi.get('pm25', {}).get('v', 0),
            'pm10': iaqi.get('pm10', {}).get('v', 0),
            'o3': iaqi.get('o3', {}).get('v', 0),
            'no2': iaqi.get('no2', {}).get('v', 0),
            'so2': iaqi.get('so2', {}).get('v', 0),
            'co': iaqi.get('co', {}).get('v', 0)
        }
        
        pollutant_fig = AQIPlotter.create_pollutant_breakdown(pollutant_data)
        st.plotly_chart(pollutant_fig, use_container_width=True)
        
        # Cartes détaillées des polluants
        cols = st.columns(3)
        
        pollutant_info = [
            ('PM2.5', 'pm25', 'μg/m³'),
            ('PM10', 'pm10', 'μg/m³'),
            ('Ozone', 'o3', 'μg/m³'),
            ('NO2', 'no2', 'μg/m³'),
            ('SO2', 'so2', 'μg/m³'),
            ('CO', 'co', 'mg/m³')
        ]
        
        for i, (name, key, unit) in enumerate(pollutant_info):
            col_idx = i % 3
            with cols[col_idx]:
                value = pollutant_data.get(key, 0)
                DashboardComponents.pollutant_mini_chart(name, value, unit)
        
        # Données météorologiques
        st.subheader("🌤️ Conditions Météorologiques")
        
        weather_data = {
            'temp': iaqi.get('t', {}).get('v'),
            'humidity': iaqi.get('h', {}).get('v'),
            'pressure': iaqi.get('p', {}).get('v'),
            'wind_speed': iaqi.get('w', {}).get('v')
        }
        
        DashboardComponents.weather_summary_card(weather_data)
    
    def display_trends_analysis(self, city: str):
        """Affiche l'analyse de tendances"""
        st.subheader("📈 Analyse de Tendances")
        
        # Génération de données historiques simulées pour la démo
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Simulation de données avec patterns réalistes
        base_trend = np.linspace(80, 120, 30)  # Tendance générale
        seasonal = 20 * np.sin(2 * np.pi * np.arange(30) / 7)  # Variation hebdomadaire
        noise = np.random.normal(0, 10, 30)  # Bruit
        
        historical_aqi = base_trend + seasonal + noise
        historical_aqi = np.clip(historical_aqi, 10, 300)  # Limiter les valeurs
        
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'aqi': historical_aqi,
            'pm25': historical_aqi * 0.6 + np.random.normal(0, 5, 30),
            'pm10': historical_aqi * 0.8 + np.random.normal(0, 8, 30),
            'hour': [d.hour for d in dates],
            'day_of_week': [d.dayofweek for d in dates]
        })
        
        # Graphique de tendance
        trend_fig = AQIPlotter.create_trend_analysis(historical_data)
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Statistiques de tendance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_aqi = historical_data['aqi'].mean()
            st.metric("AQI Moyen (30j)", f"{avg_aqi:.1f}")
        
        with col2:
            max_aqi = historical_data['aqi'].max()
            max_date = historical_data.loc[historical_data['aqi'].idxmax(), 'timestamp']
            st.metric("AQI Maximum", f"{max_aqi:.1f}")
            st.caption(f"Le {max_date.strftime('%d/%m')}")
        
        with col3:
            # Calcul de la tendance
            slope = np.polyfit(range(len(historical_data)), historical_data['aqi'], 1)[0]
            trend_direction = "📈 Hausse" if slope > 0 else "📉 Baisse" if slope < 0 else "➡️ Stable"
            st.metric("Tendance", trend_direction)
            st.caption(f"{slope:+.2f} points/jour")
    
    def display_city_comparison(self, selected_city: str):
        """Affiche la comparaison entre villes"""
        st.subheader("🌍 Comparaison Mondiale")
        
        # Villes pour la comparaison
        comparison_cities = ['barcelona', 'madrid', 'paris', 'london', 'berlin', 'rome']
        
        # Chargement des données (simulées pour la démo)
        cities_data = {}
        
        with st.spinner("Chargement des données des villes..."):
            for city in comparison_cities:
                # Simulation des données AQI
                base_aqi = np.random.uniform(30, 200)
                cities_data[city] = {'aqi': base_aqi}
        
        # Graphique de comparaison
        aqi_values = {city: data['aqi'] for city, data in cities_data.items()}
        comparison_fig = AQIPlotter.create_comparison_chart(aqi_values)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Carte des villes
        if st.checkbox("Afficher sur la carte"):
            map_fig = MapVisualizations.create_city_map(cities_data)
            st.plotly_chart(map_fig, use_container_width=True)
        
        # Classement des villes
        sorted_cities = sorted(aqi_values.items(), key=lambda x: x[1])
        
        st.subheader("🏆 Classement par Qualité de l'Air")
        
        for i, (city, aqi) in enumerate(sorted_cities):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            category = AQIUtils.get_aqi_category(aqi)
            color = AQIUtils.get_aqi_color(aqi)
            
            highlight = "**" if city == selected_city else ""
            
            st.markdown(f"""
            {medal} {highlight}{city.capitalize()}{highlight} - 
            <span style="color: {color}; font-weight: bold;">{aqi:.0f}</span> 
            ({category})
            """, unsafe_allow_html=True)
    
    def run_auto_refresh(self):
        """Gère le rafraîchissement automatique"""
        if st.session_state.auto_refresh:
            # Rafraîchissement toutes les 5 minutes
            time.sleep(300)
            st.rerun()
    
    def main(self):
        """Fonction principale de l'application"""
        # En-tête
        self.create_header()
        
        # Barre latérale et configuration
        config = self.create_sidebar()
        selected_city = config['selected_city']
        
        # Chargement des données
        with st.spinner(f"Chargement des données pour {selected_city}..."):
            current_data = self.load_current_data(selected_city)
            predictions = self.load_predictions(selected_city)
        
        # Mise à jour du timestamp
        st.session_state.last_update = datetime.now()
        
        # Affichage principal
        try:
            # Statut actuel
            self.display_current_status(current_data, selected_city)
            st.divider()
            
            # Prédictions
            if config['show_predictions']:
                self.display_predictions(predictions, selected_city)
                st.divider()
            
            # Détails des polluants
            if config['show_pollutants']:
                self.display_pollutant_details(current_data)
                st.divider()
            
            # Analyse de tendances
            if config['show_trends']:
                self.display_trends_analysis(selected_city)
                st.divider()
            
            # Comparaison de villes
            if config['show_map']:
                self.display_city_comparison(selected_city)
        
        except Exception as e:
            st.error(f"Erreur lors de l'affichage: {str(e)}")
            st.exception(e)
        
        # Footer stylé
        self.display_footer()

    def display_footer(self):
        """Affiche un pied de page parfaitement centré et harmonisé"""
        year = datetime.now().year
        st.markdown(f"""
        <style>
            .footer-container {{
                background: linear-gradient(90deg, #00416A 0%, #E4E5E6 100%);
                color: white;
                text-align: center;
                padding: 1.3rem 0;
                border-radius: 18px;
                box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
                margin-top: 3rem;
                font-size: 0.95rem;
                font-weight: 400;
                letter-spacing: 0.3px;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: fadeIn 0.8s ease-in-out;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .footer-container span {{
                font-weight: 500;
                font-size: 1rem;
            }}
        </style>
        <div class="footer-container">
            <span>© {year} 🌍 <b>AQI Predictor</b> — Tous droits réservés</span>
        </div>
        """, unsafe_allow_html=True)



# Point d'entrée de l'application
if __name__ == "__main__":
    # Vérification des variables d'environnement
# Statut de connexion
    if AQICN_API_KEY != 'demo':
        st.success("✅ API AQICN connectée - Données en temps réel")
    else:
        st.info("ℹ️ Mode simulation")
    
    # Lancement de l'application
    app = AQIStreamlitApp()
    app.main()
    
    # Rafraîchissement automatique si activé
    if st.session_state.get('auto_refresh', False):
        time.sleep(300)  # 5 minutes
        st.rerun()


