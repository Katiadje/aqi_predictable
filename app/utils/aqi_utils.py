"""
Fonctions utilitaires pour l'application AQI Prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import streamlit as st

class AQIUtils:
    """Classe utilitaire pour les calculs et transformations AQI"""
    
    # Définition des seuils AQI selon EPA
    AQI_BREAKPOINTS = {
        'pm25': [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), 
                 (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500, 301, 500)],
        'pm10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
                 (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
        'o3': [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
               (86, 105, 151, 200), (106, 200, 201, 300)]
    }
    
    # Couleurs pour chaque catégorie AQI
    AQI_COLORS = {
        'Bon': '#00E400',
        'Modéré': '#FFFF00', 
        'Malsain pour groupes sensibles': '#FF7E00',
        'Malsain': '#FF0000',
        'Très malsain': '#8F3F97',
        'Dangereux': '#7E0023'
    }
    
    # Messages de santé selon catégorie AQI
    HEALTH_MESSAGES = {
        'Bon': {
            'message': 'Qualité de l\'air excellente',
            'advice': 'Profitez de vos activités extérieures!',
            'icon': '😊'
        },
        'Modéré': {
            'message': 'Qualité de l\'air acceptable',
            'advice': 'Activités normales possibles pour la plupart des personnes',
            'icon': '😐'
        },
        'Malsain pour groupes sensibles': {
            'message': 'Risques pour personnes sensibles',
            'advice': 'Personnes avec problèmes respiratoires: limitez activités extérieures',
            'icon': '😷'
        },
        'Malsain': {
            'message': 'Risques pour la santé',
            'advice': 'Limitez les activités extérieures prolongées',
            'icon': '🤢'
        },
        'Très malsain': {
            'message': 'Risques graves pour la santé',
            'advice': 'Évitez les activités extérieures',
            'icon': '🚨'
        },
        'Dangereux': {
            'message': 'Urgence sanitaire',
            'advice': 'Restez à l\'intérieur, fermez fenêtres',
            'icon': '☠️'
        }
    }
    
    @staticmethod
    def calculate_aqi_from_pollutant(concentration: float, pollutant: str) -> float:
        """Calcule l'AQI pour un polluant donné"""
        if pollutant not in AQIUtils.AQI_BREAKPOINTS:
            return 0
        
        breakpoints = AQIUtils.AQI_BREAKPOINTS[pollutant]
        
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= concentration <= bp_high:
                # Formule EPA: AQI = ((AQI_hi - AQI_lo) / (BP_hi - BP_lo)) * (C - BP_lo) + AQI_lo
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
                return round(aqi)
        
        # Si concentration dépasse tous les seuils
        return 500
    
    @staticmethod
    def get_aqi_category(aqi_value: float) -> str:
        """Retourne la catégorie AQI basée sur la valeur"""
        if aqi_value <= 50:
            return "Bon"
        elif aqi_value <= 100:
            return "Modéré"
        elif aqi_value <= 150:
            return "Malsain pour groupes sensibles"
        elif aqi_value <= 200:
            return "Malsain"
        elif aqi_value <= 300:
            return "Très malsain"
        else:
            return "Dangereux"
    
    @staticmethod
    def get_aqi_color(aqi_value: float) -> str:
        """Retourne la couleur correspondant à la valeur AQI"""
        category = AQIUtils.get_aqi_category(aqi_value)
        return AQIUtils.AQI_COLORS.get(category, '#CCCCCC')
    
    @staticmethod
    def get_health_info(aqi_value: float) -> Dict:
        """Retourne les informations de santé pour une valeur AQI"""
        category = AQIUtils.get_aqi_category(aqi_value)
        return AQIUtils.HEALTH_MESSAGES.get(category, {
            'message': 'Données non disponibles',
            'advice': 'Consultez les autorités locales',
            'icon': '❓'
        })
    
    @staticmethod
    def format_timestamp(timestamp: datetime) -> str:
        """Formate un timestamp pour l'affichage"""
        return timestamp.strftime("%d/%m/%Y %H:%M")
    
    @staticmethod
    def validate_aqi_value(value: float) -> bool:
        """Valide qu'une valeur AQI est dans la plage acceptable"""
        return 0 <= value <= 500
    
    @staticmethod
    def interpolate_missing_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Interpole les valeurs manquantes dans un DataFrame"""
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df_cleaned.columns:
                # Interpolation linéaire pour les valeurs manquantes
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear')
                # Remplacement des valeurs restantes par la médiane
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
        return df_cleaned
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Crée des features temporelles à partir d'un timestamp"""
        df_with_time = df.copy()
        
        if timestamp_col in df.columns:
            df_with_time['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
            df_with_time['day_of_week'] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
            df_with_time['month'] = pd.to_datetime(df[timestamp_col]).dt.month
            df_with_time['is_weekend'] = df_with_time['day_of_week'].isin([5, 6])
            df_with_time['season'] = df_with_time['month'].apply(AQIUtils._get_season)
        
        return df_with_time
    
    @staticmethod
    def _get_season(month: int) -> int:
        """Détermine la saison basée sur le mois"""
        if month in [12, 1, 2]:
            return 0  # Hiver
        elif month in [3, 4, 5]:
            return 1  # Printemps
        elif month in [6, 7, 8]:
            return 2  # Été
        else:
            return 3  # Automne

class DataValidator:
    """Classe pour valider les données AQI"""
    
    @staticmethod
    def validate_pollutant_data(data: Dict) -> Tuple[bool, List[str]]:
        """Valide les données de polluants"""
        errors = []
        
        # Vérification des polluants principaux
        required_fields = ['pm25', 'pm10', 'o3', 'no2']
        for field in required_fields:
            if field not in data:
                errors.append(f"Champ manquant: {field}")
            elif not isinstance(data[field], (int, float)):
                errors.append(f"Type invalide pour {field}")
            elif data[field] < 0:
                errors.append(f"Valeur négative pour {field}")
        
        # Vérification de cohérence PM2.5 vs PM10
        if 'pm25' in data and 'pm10' in data:
            if data['pm25'] > data['pm10']:
                errors.append("PM2.5 ne peut pas être supérieur à PM10")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_weather_data(data: Dict) -> Tuple[bool, List[str]]:
        """Valide les données météorologiques"""
        errors = []
        
        # Validation température
        if 'temp' in data:
            if not -50 <= data['temp'] <= 60:
                errors.append("Température hors limites réalistes (-50°C à 60°C)")
        
        # Validation humidité
        if 'humidity' in data:
            if not 0 <= data['humidity'] <= 100:
                errors.append("Humidité doit être entre 0% et 100%")
        
        # Validation pression
        if 'pressure' in data:
            if not 800 <= data['pressure'] <= 1200:
                errors.append("Pression atmosphérique hors limites réalistes")
        
        # Validation vitesse du vent
        if 'wind_speed' in data:
            if data['wind_speed'] < 0 or data['wind_speed'] > 200:
                errors.append("Vitesse du vent invalide")
        
        return len(errors) == 0, errors

class APIHelper:
    """Classe pour interagir avec les APIs externes"""
    
    @staticmethod
    def fetch_city_aqi(city: str, api_key: str = None) -> Optional[Dict]:
        """Récupère les données AQI pour une ville donnée"""
        try:
            api_key = api_key or 'demo'
            url = f"https://api.aqicn.org/feed/{city}/?token={api_key}"
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'ok':
                return data['data']
            else:
                st.error(f"Erreur API: {data.get('msg', 'Erreur inconnue')}")
                return APIHelper._generate_fallback_data(city)
                
        except requests.exceptions.Timeout:
            st.error("Timeout lors de la requête API")
            return APIHelper._generate_fallback_data(city)
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur réseau: {str(e)}")
            return APIHelper._generate_fallback_data(city)
        except Exception as e:
            st.error(f"Erreur inattendue: {str(e)}")
            return APIHelper._generate_fallback_data(city)
    
    @staticmethod
    def _generate_fallback_data(city: str) -> Dict:
        """Génère des données de fallback réalistes"""
        # Patterns de base par ville
        city_patterns = {
            'barcelona': {'base_aqi': 75, 'temp': 22, 'humidity': 65},
            'paris': {'base_aqi': 85, 'temp': 18, 'humidity': 70},
            'london': {'base_aqi': 65, 'temp': 15, 'humidity': 80},
            'madrid': {'base_aqi': 90, 'temp': 20, 'humidity': 60},
            'berlin': {'base_aqi': 70, 'temp': 16, 'humidity': 75},
            'rome': {'base_aqi': 80, 'temp': 24, 'humidity': 68}
        }
        
        pattern = city_patterns.get(city.lower(), city_patterns['paris'])
        
        # Variation réaliste selon l'heure
        current_hour = datetime.now().hour
        hour_factor = 1.2 if current_hour in [7, 8, 9, 17, 18, 19] else 0.9
        random_factor = np.random.uniform(0.8, 1.3)
        
        aqi = max(20, min(200, pattern['base_aqi'] * hour_factor * random_factor))
        
        return {
            'aqi': int(aqi),
            'iaqi': {
                'pm25': {'v': max(5, aqi * 0.6 + np.random.normal(0, 5))},
                'pm10': {'v': max(8, aqi * 0.8 + np.random.normal(0, 8))},
                'o3': {'v': max(10, aqi * 0.4 + np.random.normal(0, 10))},
                'no2': {'v': max(5, aqi * 0.3 + np.random.normal(0, 8))},
                'so2': {'v': max(2, aqi * 0.2 + np.random.normal(0, 5))},
                'co': {'v': max(1, aqi * 0.1 + np.random.normal(0, 3))},
                't': {'v': pattern['temp'] + np.random.normal(0, 3)},
                'h': {'v': max(20, min(95, pattern['humidity'] + np.random.normal(0, 10)))},
                'p': {'v': np.random.normal(1013, 8)},
                'w': {'v': max(0, np.random.normal(4, 2))}
            },
            'time': {'s': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            '_source': 'fallback_simulation'
        }
    
    @staticmethod
    def get_available_cities() -> List[str]:
        """Retourne la liste des villes disponibles"""
        return [
            'barcelona', 'madrid', 'paris', 'london', 'berlin',
            'rome', 'amsterdam', 'brussels', 'vienna', 'zurich',
            'lisbon', 'dublin', 'stockholm', 'helsinki', 'oslo',
            'copenhagen', 'warsaw', 'prague', 'budapest', 'athens'
        ]
    
    @staticmethod
    def validate_city(city: str) -> bool:
        """Valide qu'une ville est disponible"""
        return city.lower() in [c.lower() for c in APIHelper.get_available_cities()]

class CacheManager:
    """Gestionnaire de cache pour optimiser les performances"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache pendant 1 heure
    def cached_api_call(city: str, api_key: str) -> Optional[Dict]:
        """Appel API avec cache"""
        return APIHelper.fetch_city_aqi(city, api_key)
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache pendant 5 minutes
    def cached_predictions(city: str) -> Dict:
        """Cache des prédictions"""
        # Cette fonction sera implémentée avec le pipeline d'inférence
        return {}
    
    @staticmethod
    def clear_cache():
        """Vide le cache Streamlit"""
        st.cache_data.clear()

class MetricsCalculator:
    """Calculateur de métriques pour l'évaluation des modèles"""
    
    @staticmethod
    def calculate_accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calcule les métriques de précision"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Métriques spécifiques à l'AQI
        category_accuracy = MetricsCalculator._calculate_category_accuracy(y_true, y_pred)
        
        return {
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 3),
            'category_accuracy': round(category_accuracy, 3)
        }
    
    @staticmethod
    def _calculate_category_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule la précision de classification par catégorie AQI"""
        true_categories = [AQIUtils.get_aqi_category(aqi) for aqi in y_true]
        pred_categories = [AQIUtils.get_aqi_category(aqi) for aqi in y_pred]
        
        correct = sum(1 for true_cat, pred_cat in zip(true_categories, pred_categories) 
                     if true_cat == pred_cat)
        
        return correct / len(y_true) if len(y_true) > 0 else 0.0