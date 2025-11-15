import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import requests
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class AQIInferencePipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def connect_to_hopsworks(self):
        """Se connecte à Hopsworks (simulé pour la démo)"""
        try:
            print("⚠️ Mode démo - Hopsworks simulé")
            return True
        except Exception as e:
            print(f"❌ Erreur de connexion à Hopsworks: {e}")
            return False
    
    def load_model(self):
        """Charge le modèle (simulé si pas trouvé)"""
        try:
            # Essai de chargement local
            if os.path.exists("aqi_model/model.pkl") and os.path.exists("aqi_model/scaler.pkl"):
                self.model = joblib.load("aqi_model/model.pkl")
                self.scaler = joblib.load("aqi_model/scaler.pkl")
                print("✅ Modèle local chargé avec succès")
                return True
            else:
                print("⚠️ Pas de modèle local trouvé, simulation des prédictions")
                return True
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du modèle: {e}")
            print("🔄 Mode simulation activé")
            return True
    
    def get_latest_features(self, city: str = "paris") -> pd.DataFrame:
        """Récupère les dernières features (simulées)"""
        try:
            print(f"📊 Génération de features simulées pour {city}")
            
            # Génération de données simulées réalistes
            current_time = datetime.now()
            
            # Base AQI avec variation réaliste selon la ville
            city_base_aqi = {
                'paris': 85,
                'barcelona': 75,
                'london': 65,
                'madrid': 90,
                'berlin': 70,
                'rome': 80
            }
            
            base_aqi = city_base_aqi.get(city.lower(), 80) + np.random.normal(0, 15)
            base_aqi = max(10, min(300, base_aqi))
            
            data = {
                'timestamp': current_time,
                'city': city,
                'aqi': base_aqi,
                'pm25': max(0, base_aqi * 0.6 + np.random.normal(0, 10)),
                'pm10': max(0, base_aqi * 0.8 + np.random.normal(0, 15)),
                'o3': max(0, base_aqi * 0.4 + np.random.normal(0, 8)),
                'no2': max(0, base_aqi * 0.3 + np.random.normal(0, 5)),
                'so2': max(0, base_aqi * 0.2 + np.random.normal(0, 3)),
                'co': max(0, base_aqi * 0.1 + np.random.normal(0, 2)),
                'temp': np.random.normal(18, 6),  # Température parisienne
                'humidity': np.random.normal(65, 15),
                'pressure': np.random.normal(1013, 10),
                'wind_speed': max(0, np.random.normal(4, 2)),
                'hour': current_time.hour,
                'day_of_week': current_time.weekday(),
                'month': current_time.month,
                'is_weekend': current_time.weekday() >= 5,
                'season': self._get_season(current_time.month)
            }
            
            # Features dérivées
            data['pm_ratio'] = data['pm25'] / max(data['pm10'], 1)
            data['pollution_score'] = (data['pm25'] * 0.4 + data['pm10'] * 0.3 + 
                                     data['o3'] * 0.2 + data['no2'] * 0.1)
            data['temp_humidity_index'] = data['temp'] * data['humidity'] / 100
            
            df = pd.DataFrame([data])
            print(f"📊 {len(df)} enregistrements simulés générés")
            return df
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération des features: {e}")
            return pd.DataFrame()
    
    def fetch_current_aqi_data(self, city: str = "paris") -> Dict:
        """Récupère les données AQI actuelles depuis l'API avec timeout augmenté"""
        try:
            api_key = os.getenv('AQICN_API_KEY', 'demo')
            url = f"https://api.aqicn.org/feed/{city}/?token={api_key}"
            
            print(f"🔗 Tentative de connexion à l'API pour {city}...")
            
            # Timeout augmenté à 30 secondes
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'ok':
                print(f"✅ Données API récupérées pour {city}")
                aqi_value = data['data'].get('aqi', 'N/A')
                print(f"📊 AQI {city}: {aqi_value}")
                return data['data']
            else:
                print(f"⚠️ Erreur API: {data}")
                return self._get_fallback_data(city)
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout API pour {city} (>30s), utilisation de données de secours")
            return self._get_fallback_data(city)
        except requests.exceptions.RequestException as e:
            print(f"🌐 Erreur réseau pour {city}: {e}")
            return self._get_fallback_data(city)
        except Exception as e:
            print(f"⚠️ Erreur inattendue pour {city}: {e}")
            return self._get_fallback_data(city)
    
    def _get_fallback_data(self, city: str) -> Dict:
        """Données de secours réalistes par ville"""
        print(f"🔄 Génération de données de secours pour {city}")
        
        # Données réalistes par ville
        city_data = {
            'paris': {
                'aqi': np.random.randint(70, 120),
                'pm25': np.random.randint(15, 45),
                'pm10': np.random.randint(25, 65),
                'o3': np.random.randint(20, 80),
                'no2': np.random.randint(30, 70),
                'temp': np.random.randint(12, 25),
                'humidity': np.random.randint(50, 85)
            },
            'barcelona': {
                'aqi': np.random.randint(60, 110),
                'pm25': np.random.randint(12, 40),
                'pm10': np.random.randint(20, 60),
                'o3': np.random.randint(25, 90),
                'no2': np.random.randint(25, 65),
                'temp': np.random.randint(16, 28),
                'humidity': np.random.randint(45, 80)
            },
            'london': {
                'aqi': np.random.randint(50, 100),
                'pm25': np.random.randint(10, 35),
                'pm10': np.random.randint(18, 55),
                'o3': np.random.randint(15, 70),
                'no2': np.random.randint(35, 75),
                'temp': np.random.randint(8, 20),
                'humidity': np.random.randint(60, 90)
            }
        }
        
        base_data = city_data.get(city.lower(), city_data['paris'])
        
        return {
            'aqi': base_data['aqi'],
            'iaqi': {
                'pm25': {'v': base_data['pm25']},
                'pm10': {'v': base_data['pm10']},
                'o3': {'v': base_data['o3']},
                'no2': {'v': base_data['no2']},
                'so2': {'v': np.random.randint(5, 25)},
                'co': {'v': np.random.randint(2, 15)},
                't': {'v': base_data['temp']},
                'h': {'v': base_data['humidity']},
                'p': {'v': np.random.randint(1005, 1025)},
                'w': {'v': np.random.randint(2, 12)}
            },
            'time': {
                's': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def create_future_features(self, base_features: pd.DataFrame, hours_ahead: int) -> pd.DataFrame:
        """Crée des features pour les prédictions futures"""
        if base_features.empty:
            return pd.DataFrame()
        
        # Utilise les dernières features comme base
        latest_features = base_features.iloc[-1].copy()
        
        # Calcul du timestamp futur
        current_time = datetime.now()
        future_time = current_time + timedelta(hours=hours_ahead)
        
        # Mise à jour des features temporelles
        latest_features['timestamp'] = future_time
        latest_features['hour'] = future_time.hour
        latest_features['day_of_week'] = future_time.weekday()
        latest_features['month'] = future_time.month
        latest_features['is_weekend'] = future_time.weekday() >= 5
        latest_features['season'] = self._get_season(future_time.month)
        
        # Ajout de variations réalistes basées sur l'heure et la météo
        hour_factor = np.sin(2 * np.pi * future_time.hour / 24) * 0.2 + 1
        
        # Variation saisonnière (plus de pollution en hiver)
        seasonal_factor = 1.2 if future_time.month in [11, 12, 1, 2] else 0.9
        
        # Simulation de variations légères pour les polluants
        pollution_vars = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        for var in pollution_vars:
            if var in latest_features:
                base_value = latest_features[var]
                # Variation de ±15% avec facteurs horaire et saisonnier
                variation = np.random.normal(0, 0.15) * base_value * hour_factor * seasonal_factor
                latest_features[var] = max(0, base_value + variation)
        
        # Recalcul des features dérivées
        latest_features['pm_ratio'] = latest_features['pm25'] / max(latest_features['pm10'], 1)
        latest_features['pollution_score'] = (
            latest_features['pm25'] * 0.4 + 
            latest_features['pm10'] * 0.3 + 
            latest_features['o3'] * 0.2 + 
            latest_features['no2'] * 0.1
        )
        latest_features['temp_humidity_index'] = latest_features['temp'] * latest_features['humidity'] / 100
        
        return pd.DataFrame([latest_features])
    
    def _get_season(self, month: int) -> int:
        """Détermine la saison basée sur le mois"""
        if month in [12, 1, 2]:
            return 0  # Hiver
        elif month in [3, 4, 5]:
            return 1  # Printemps
        elif month in [6, 7, 8]:
            return 2  # Été
        else:
            return 3  # Automne
    
    def prepare_features_for_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """Prépare les features pour la prédiction"""
        feature_cols = [
            'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
            'temp', 'humidity', 'pressure', 'wind_speed',
            'hour', 'day_of_week', 'month', 'is_weekend', 'season',
            'pm_ratio', 'pollution_score', 'temp_humidity_index'
        ]
        
        # Sélection et remplissage des features
        X = df[feature_cols].fillna(0)
        
        # Si on a un scaler, l'utiliser, sinon simuler
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            # Normalisation simple pour simulation
            X_scaled = (X - X.mean()) / (X.std() + 1e-8)
            X_scaled = X_scaled.fillna(0).values
        
        return X_scaled
    
    def predict_aqi(self, city: str = "paris", days_ahead: int = 3) -> List[Dict]:
        """Génère les prédictions AQI pour les prochains jours"""
        predictions = []
        
        try:
            # Récupération des features récentes
            recent_features = self.get_latest_features(city)
            
            if recent_features.empty:
                print("⚠️ Aucune donnée historique disponible")
                return []
            
            # Génération des prédictions pour chaque jour
            for day in range(days_ahead):
                daily_predictions = []
                
                # Prédictions toutes les 6 heures pour chaque jour
                for hour_offset in [6, 12, 18, 24]:
                    hours_ahead = (day * 24) + hour_offset
                    
                    # Création des features futures
                    future_features = self.create_future_features(recent_features, hours_ahead)
                    
                    if future_features.empty:
                        continue
                    
                    # Prédiction (simulée si pas de modèle)
                    if self.model:
                        X = self.prepare_features_for_prediction(future_features)
                        aqi_pred = self.model.predict(X)[0]
                    else:
                        # Simulation de prédiction basée sur les tendances et patterns réalistes
                        base_aqi = recent_features['aqi'].iloc[0]
                        
                        # Facteur horaire (plus pollué aux heures de pointe)
                        hour = (hours_ahead % 24)
                        if hour in [7, 8, 9, 17, 18, 19]:  # Heures de pointe
                            hour_factor = 1.2
                        elif hour in [2, 3, 4, 5]:  # Nuit
                            hour_factor = 0.8
                        else:
                            hour_factor = 1.0
                        
                        # Facteur jour de semaine (plus pollué en semaine)
                        day_of_week = (datetime.now() + timedelta(hours=hours_ahead)).weekday()
                        week_factor = 1.1 if day_of_week < 5 else 0.9
                        
                        # Tendance générale avec un peu de randomness
                        trend_factor = np.random.normal(0, 8)
                        
                        # Prédiction finale
                        aqi_pred = base_aqi * hour_factor * week_factor + trend_factor
                    
                    aqi_pred = max(15, min(250, aqi_pred))  # Limiter entre 15 et 250
                    
                    # Timestamp de la prédiction
                    pred_time = datetime.now() + timedelta(hours=hours_ahead)
                    
                    daily_predictions.append({
                        'timestamp': pred_time,
                        'aqi': round(aqi_pred, 1),
                        'hour': pred_time.hour
                    })
                
                # Calcul de la moyenne journalière
                if daily_predictions:
                    avg_aqi = np.mean([p['aqi'] for p in daily_predictions])
                    date = (datetime.now() + timedelta(days=day+1)).date()
                    
                    # Calcul de la moyenne et plage pour ce jour
                if daily_predictions:
                    avg_aqi = np.mean([p['aqi'] for p in daily_predictions])
                    min_aqi = min([p['aqi'] for p in daily_predictions])
                    max_aqi = max([p['aqi'] for p in daily_predictions])
                    date = (datetime.now() + timedelta(days=day+1)).date()
                    category = self._get_aqi_category(avg_aqi)
                    
                    # Générer prédictions horaires pour ce jour
                    hourly_preds = []
                    for hour in range(24):
                        hourly_aqi = avg_aqi + np.random.normal(0, 8)  # Variation horaire
                        hourly_preds.append({
                            'hour': hour,
                            'timestamp': f"{date} {hour:02d}:00",
                            'aqi': max(0, min(500, round(hourly_aqi, 1)))
                        })
                    
                    # Calcul de la moyenne et plage pour ce jour
                if daily_predictions:
                    avg_aqi = np.mean([p['aqi'] for p in daily_predictions])
                    min_aqi = min([p['aqi'] for p in daily_predictions])
                    max_aqi = max([p['aqi'] for p in daily_predictions])
                    date = (datetime.now() + timedelta(days=day+1)).date()
                    category = self._get_aqi_category(avg_aqi)
                    
                    # Générer prédictions horaires pour ce jour
                    hourly_preds = []
                    for hour in range(24):
                        hourly_aqi = avg_aqi + np.random.normal(0, 8)  # Variation horaire
                        hourly_preds.append({
                            'hour': hour,
                            'timestamp': f"{date} {hour:02d}:00",
                            'aqi': max(0, min(500, round(hourly_aqi, 1)))
                        })
                    
                    predictions.append({
                        'day': day + 1,
                        'date': date,
                        'aqi_avg': round(avg_aqi, 1),
                        'aqi_min': round(min_aqi, 1),
                        'aqi_max': round(max_aqi, 1),
                        'category': category,
                        'city': city,
                        'hourly_predictions': hourly_preds
                    })
            
            print(f"✅ {len(predictions)} prédictions générées pour {city}")
            return predictions
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return []
    
    def _get_aqi_category(self, aqi: float) -> str:
        """Retourne la catégorie AQI"""
        if aqi <= 50:
            return "Bon"
        elif aqi <= 100:
            return "Modéré"
        elif aqi <= 150:
            return "Malsain pour groupes sensibles"
        elif aqi <= 200:
            return "Malsain"
        elif aqi <= 300:
            return "Très malsain"
        else:
            return "Dangereux"
    
    def get_current_aqi_status(self, city: str = "paris") -> Dict:
        """Récupère le statut AQI actuel"""
        try:
            # Données API temps réel avec fallback robuste
            current_data = self.fetch_current_aqi_data(city)
            
            if not current_data:
                # Fallback sur les données simulées
                current_aqi = np.random.randint(60, 120)
                source = 'fallback_simulation'
            else:
                current_aqi = current_data.get('aqi', np.random.randint(60, 120))
                source = 'api' if 'iaqi' in current_data else 'fallback_realistic'
            
            return {
                'city': city,
                'current_aqi': current_aqi,
                'category': self._get_aqi_category(current_aqi),
                'timestamp': datetime.now(),
                'source': source
            }
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération du statut actuel: {e}")
            return {
                'city': city,
                'current_aqi': np.random.randint(60, 120),
                'category': 'Modéré',
                'timestamp': datetime.now(),
                'source': 'emergency_fallback'
            }
    
    def generate_forecast_report(self, predictions: List[Dict], current_status: Dict) -> str:
        """Génère un rapport de prévision"""
        if not predictions:
            return "❌ Aucune prédiction disponible"
        
        city = predictions[0]['city']
        current_aqi = current_status.get('current_aqi', 'N/A')
        source = current_status.get('source', 'unknown')
        
        # Icône selon la source
        source_icon = {
            'api': '🌐',
            'fallback_realistic': '🔄',
            'fallback_simulation': '🎲',
            'emergency_fallback': '⚠️'
        }
        
        report = f"""
# 🌍 Rapport de Prévision AQI - {city.capitalize()}

## 📊 Statut Actuel
- **AQI Actuel**: {current_aqi}
- **Catégorie**: {current_status.get('category', 'N/A')}
- **Source**: {source_icon.get(source, '❓')} {source.replace('_', ' ').title()}
- **Dernière mise à jour**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 📈 Prévisions (3 prochains jours)
"""
        
        for pred in predictions:
            date_str = pred['date'].strftime('%d/%m/%Y')
            report += f"""
### Jour {pred['day']} - {date_str}
- **AQI Moyen**: {pred['aqi_avg']} ({pred['category']})
- **Plage**: {pred['aqi_min']} - {pred['aqi_max']}
"""
        
        # Tendance générale
        trend_values = [p['aqi_avg'] for p in predictions]
        if len(trend_values) >= 2:
            if trend_values[-1] > trend_values[0]:
                trend = "📈 Dégradation prévue"
            elif trend_values[-1] < trend_values[0]:
                trend = "📉 Amélioration prévue"
            else:
                trend = "➡️ Stable"
            
            report += f"\n## 🔮 Tendance: {trend}\n"
        
        # Note sur la source des données
        if source != 'api':
            report += f"\n⚠️ **Note**: Données {source.replace('_', ' ')} utilisées (API non disponible)\n"
        
        return report
    
    def run_inference_pipeline(self, city: str = "paris") -> Dict:
        """Exécute le pipeline d'inférence complet"""
        print(f"🚀 Démarrage du pipeline d'inférence pour {city}...")
        
        # Connexion simulée
        if not self.connect_to_hopsworks():
            return {"error": "Connexion échouée", "success": False}
        
        # Chargement du modèle (avec fallback)
        if not self.load_model():
            return {"error": "Chargement du modèle échoué", "success": False}
        
        # Statut actuel
        current_status = self.get_current_aqi_status(city)
        
        # Génération des prédictions
        predictions = self.predict_aqi(city, days_ahead=3)
        
        if not predictions:
            return {"error": "Aucune prédiction générée", "success": False}
        
        # Rapport
        report = self.generate_forecast_report(predictions, current_status)
        
        result = {
            "city": city,
            "timestamp": datetime.now().isoformat(),
            "current_status": current_status,
            "predictions": predictions,
            "report": report,
            "success": True,
            "mode": "resilient_demo"
        }
        
        print("✅ Pipeline d'inférence terminé avec succès (mode résilient)")
        return result

if __name__ == "__main__":
    import sys
    
    # Récupération de la ville depuis les arguments
    city = sys.argv[1] if len(sys.argv) > 1 else "paris"
    
    # Création et exécution du pipeline
    inference_pipeline = AQIInferencePipeline()
    result = inference_pipeline.run_inference_pipeline(city)
    
    if result.get("success"):
        print("\n" + result["report"])
    else:
        print(f"❌ Erreur: {result.get('error')}")