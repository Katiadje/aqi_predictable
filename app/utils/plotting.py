"""
Fonctions de visualisation pour l'application AQI Prediction
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import streamlit as st
from app.utils.aqi_utils import AQIUtils

class AQIPlotter:
    """Classe pour créer des visualisations AQI"""
    
    @staticmethod
    def create_aqi_gauge(current_aqi: float, title: str = "AQI Actuel") -> go.Figure:
        """Crée un gauge pour afficher l'AQI actuel"""
        color = AQIUtils.get_aqi_color(current_aqi)
        category = AQIUtils.get_aqi_category(current_aqi)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_aqi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{title}<br><span style='font-size:0.8em;color:gray'>{category}</span>"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 500]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': '#00E400'},
                    {'range': [50, 100], 'color': '#FFFF00'},
                    {'range': [100, 150], 'color': '#FF7E00'},
                    {'range': [150, 200], 'color': '#FF0000'},
                    {'range': [200, 300], 'color': '#8F3F97'},
                    {'range': [300, 500], 'color': '#7E0023'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 200
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor="white"
        )
        
        return fig
    
    @staticmethod
    def create_prediction_timeline(predictions: List[Dict], city: str) -> go.Figure:
        """Crée un graphique temporel des prédictions"""
        if not predictions:
            return go.Figure().add_annotation(text="Aucune prédiction disponible")
        
        # Préparation des données
        dates = [pred['date'] for pred in predictions]
        aqi_avg = [pred['aqi_avg'] for pred in predictions]
        aqi_min = [pred['aqi_min'] for pred in predictions]
        aqi_max = [pred['aqi_max'] for pred in predictions]
        categories = [pred['category'] for pred in predictions]
        
        # Couleurs pour chaque point
        colors = [AQIUtils.get_aqi_color(aqi) for aqi in aqi_avg]
        
        fig = go.Figure()
        
        # Zone d'incertitude (min-max)
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=aqi_max + aqi_min[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Plage de prédiction'
        ))
        
        # Ligne de prédiction moyenne
        fig.add_trace(go.Scatter(
            x=dates,
            y=aqi_avg,
            mode='lines+markers',
            name='AQI Prédit',
            line=dict(color='rgb(31, 119, 180)', width=3),
            marker=dict(
                size=12,
                color=colors,
                line=dict(width=2, color='white')
            ),
            text=categories,
            hovertemplate='<b>%{x}</b><br>AQI: %{y}<br>Catégorie: %{text}<extra></extra>'
        ))
        
        # Lignes de seuils AQI
        fig.add_hline(y=50, line_dash="dash", line_color="green", 
                     annotation_text="Bon", annotation_position="bottom right")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", 
                     annotation_text="Modéré", annotation_position="bottom right")
        fig.add_hline(y=150, line_dash="dash", line_color="orange", 
                     annotation_text="Malsain (sensibles)", annotation_position="bottom right")
        fig.add_hline(y=200, line_dash="dash", line_color="red", 
                     annotation_text="Malsain", annotation_position="bottom right")
        
        fig.update_layout(
            title=f'🔮 Prédictions AQI - {city.capitalize()}',
            xaxis_title='Date',
            yaxis_title='Indice AQI',
            height=500,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_pollutant_breakdown(pollutant_data: Dict) -> go.Figure:
        """Crée un graphique en barres des polluants"""
        pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
        values = [
            pollutant_data.get('pm25', 0),
            pollutant_data.get('pm10', 0),
            pollutant_data.get('o3', 0),
            pollutant_data.get('no2', 0),
            pollutant_data.get('so2', 0),
            pollutant_data.get('co', 0)
        ]
        
        # Couleurs dégradées
        colors = px.colors.sequential.Reds_r[:len(pollutants)]
        
        fig = go.Figure(data=[
            go.Bar(
                x=pollutants,
                y=values,
                marker_color=colors,
                text=[f'{v:.1f}' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Concentration: %{y}<br><extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='🏭 Répartition des Polluants',
            xaxis_title='Polluants',
            yaxis_title='Concentration (μg/m³)',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_hourly_forecast(predictions: List[Dict]) -> go.Figure:
        """Crée un graphique des prévisions horaires détaillées"""
        if not predictions:
            return go.Figure()
        
        # Collecte de toutes les prédictions horaires
        all_hourly = []
        for day_pred in predictions:
            for hourly in day_pred.get('hourly_predictions', []):
                all_hourly.append({
                    'timestamp': hourly['timestamp'],
                    'aqi': hourly['aqi'],
                    'hour': hourly['hour'],
                    'day': day_pred['day']
                })
        
        if not all_hourly:
            return go.Figure()
        
        df_hourly = pd.DataFrame(all_hourly)
        
        # Couleurs par jour
        day_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
        
        fig = go.Figure()
        
        for day in df_hourly['day'].unique():
            day_data = df_hourly[df_hourly['day'] == day]
            
            fig.add_trace(go.Scatter(
                x=day_data['timestamp'],
                y=day_data['aqi'],
                mode='lines+markers',
                name=f'Jour {day}',
                line=dict(color=day_colors.get(day, '#333333'), width=2),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>AQI: %{y}<br><extra></extra>'
            ))
        
        fig.update_layout(
            title='⏰ Prévisions Horaires Détaillées',
            xaxis_title='Heure',
            yaxis_title='AQI',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_comparison_chart(cities_data: Dict[str, float]) -> go.Figure:
        """Crée un graphique de comparaison entre villes"""
        cities = list(cities_data.keys())
        aqi_values = list(cities_data.values())
        colors = [AQIUtils.get_aqi_color(aqi) for aqi in aqi_values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=cities,
                y=aqi_values,
                marker_color=colors,
                text=[f'{v:.0f}' for v in aqi_values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>AQI: %{y}<br><extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='🌍 Comparaison AQI par Ville',
            xaxis_title='Villes',
            yaxis_title='AQI',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_trend_analysis(historical_data: pd.DataFrame) -> go.Figure:
        """Crée une analyse de tendance des données historiques"""
        if historical_data.empty:
            return go.Figure()
        
        # Préparation des données
        df = historical_data.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Évolution AQI', 'Distribution PM2.5', 'Moyennes par Heure', 'Corrélations'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Évolution temporelle AQI
        if 'aqi' in df.columns and 'timestamp' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['aqi'], name='AQI', 
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # 2. Distribution PM2.5
        if 'pm25' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['pm25'], name='PM2.5', marker_color='red', opacity=0.7),
                row=1, col=2
            )
        
        # 3. Moyennes par heure
        if 'hour' in df.columns and 'aqi' in df.columns:
            hourly_avg = df.groupby('hour')['aqi'].mean()
            fig.add_trace(
                go.Bar(x=hourly_avg.index, y=hourly_avg.values, name='AQI Moy/Heure',
                      marker_color='green'),
                row=2, col=1
            )
        
        # 4. Heatmap corrélations (simplifié)
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limiter à 5 colonnes
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', showscale=False),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="📊 Analyse de Tendance des Données Historiques",
            showlegend=False,
            template='plotly_white'
        )
        
        return fig

class DashboardComponents:
    """Composants réutilisables pour le dashboard"""
    
    @staticmethod
    def aqi_metric_card(aqi_value: float, title: str, delta: Optional[float] = None):
        """Affiche une carte métrique AQI"""
        category = AQIUtils.get_aqi_category(aqi_value)
        color = AQIUtils.get_aqi_color(aqi_value)
        health_info = AQIUtils.get_health_info(aqi_value)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric(
                label=title,
                value=f"{aqi_value:.0f}",
                delta=f"{delta:+.1f}" if delta is not None else None
            )
            st.write(f"**{category}** {health_info['icon']}")
            st.caption(health_info['message'])
        
        with col2:
            # Petit indicateur coloré
            st.markdown(f"""
            <div style="width: 30px; height: 30px; background-color: {color}; 
                        border-radius: 50%; margin: 10px auto;"></div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def pollutant_mini_chart(pollutant_name: str, value: float, unit: str = "μg/m³"):
        """Affiche un mini-graphique pour un polluant"""
        # Seuils simplifiés pour chaque polluant
        thresholds = {
            'PM2.5': [12, 35, 55, 150],
            'PM10': [54, 154, 254, 354],
            'O3': [54, 70, 85, 105],
            'NO2': [53, 100, 360, 649],
            'SO2': [35, 75, 185, 304],
            'CO': [4.4, 9.4, 12.4, 15.4]
        }
        
        threshold = thresholds.get(pollutant_name, [50, 100, 150, 200])
        
        # Détermination du niveau
        if value <= threshold[0]:
            level = "Bon"
            color = "#00E400"
        elif value <= threshold[1]:
            level = "Modéré"
            color = "#FFFF00"
        elif value <= threshold[2]:
            level = "Élevé"
            color = "#FF7E00"
        else:
            level = "Très élevé"
            color = "#FF0000"
        
        # Affichage
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**{pollutant_name}**")
            st.write(f"{value:.1f} {unit}")
        with col2:
            st.markdown(f"""
            <div style="width: 20px; height: 20px; background-color: {color}; 
                        border-radius: 3px; margin: 5px auto;"></div>
            <small style="display: block; text-align: center;">{level}</small>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def create_alert_banner(message: str, alert_type: str = "info"):
        """Crée une bannière d'alerte"""
        alert_colors = {
            "info": "#d1ecf1",
            "warning": "#fff3cd", 
            "danger": "#f8d7da",
            "success": "#d4edda"
        }
        
        alert_icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "danger": "🚨",
            "success": "✅"
        }
        
        color = alert_colors.get(alert_type, "#d1ecf1")
        icon = alert_icons.get(alert_type, "ℹ️")
        
        st.markdown(f"""
        <div style="padding: 1rem; background-color: {color}; 
                    border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #007bff;">
            <strong>{icon} {message}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def weather_summary_card(weather_data: Dict):
        """Affiche un résumé météo"""
        if not weather_data:
            st.warning("Données météo non disponibles")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp = weather_data.get('temp', 'N/A')
            st.metric("🌡️ Température", f"{temp}°C" if isinstance(temp, (int, float)) else temp)
        
        with col2:
            humidity = weather_data.get('humidity', 'N/A')
            st.metric("💧 Humidité", f"{humidity}%" if isinstance(humidity, (int, float)) else humidity)
        
        with col3:
            pressure = weather_data.get('pressure', 'N/A')
            st.metric("🌪️ Pression", f"{pressure} hPa" if isinstance(pressure, (int, float)) else pressure)
        
        with col4:
            wind = weather_data.get('wind_speed', 'N/A')
            st.metric("💨 Vent", f"{wind} km/h" if isinstance(wind, (int, float)) else wind)

class MapVisualizations:
    """Visualisations cartographiques"""
    
    @staticmethod
    def create_city_map(cities_data: Dict[str, Dict]) -> go.Figure:
        """Crée une carte avec les AQI de différentes villes"""
        # Coordonnées approximatives des villes européennes
        city_coords = {
            'barcelona': {'lat': 41.3851, 'lon': 2.1734},
            'madrid': {'lat': 40.4168, 'lon': -3.7038},
            'paris': {'lat': 48.8566, 'lon': 2.3522},
            'london': {'lat': 51.5074, 'lon': -0.1278},
            'berlin': {'lat': 52.5200, 'lon': 13.4050},
            'rome': {'lat': 41.9028, 'lon': 12.4964},
            'amsterdam': {'lat': 52.3676, 'lon': 4.9041},
        }
        
        # Préparation des données pour la carte
        lats, lons, city_names, aqi_values, colors = [], [], [], [], []
        
        for city, data in cities_data.items():
            if city.lower() in city_coords:
                coords = city_coords[city.lower()]
                aqi = data.get('aqi', 0)
                
                lats.append(coords['lat'])
                lons.append(coords['lon'])
                city_names.append(city.capitalize())
                aqi_values.append(aqi)
                colors.append(AQIUtils.get_aqi_color(aqi))
        
        # Création de la carte
        fig = go.Figure(data=go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=15,
                color=colors,
                opacity=0.8,
                sizemode='diameter'
            ),
            text=city_names,
            customdata=aqi_values,
            hovertemplate='<b>%{text}</b><br>AQI: %{customdata}<extra></extra>'
        ))
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                bearing=0,
                center=dict(lat=48, lon=5),
                pitch=0,
                zoom=4
            ),
            height=500,
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        
        return fig