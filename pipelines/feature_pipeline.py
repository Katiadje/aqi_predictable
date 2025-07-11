"""
📊 AQI Feature Pipeline
======================

Pipeline de collecte et traitement des données AQI en temps réel.

Fonctionnalités:
- Récupération données API AQICN.org
- Feature engineering automatique
- Sauvegarde Hopsworks Feature Store
- Mode backfill pour données historiques
- Validation et nettoyage des données

Usage:
    python feature_pipeline.py                    # Collecte temps réel
    python feature_pipeline.py --backfill --days=30  # Données historiques
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import argparse
from typing import Dict, Optional, Tuple, Any
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Suppression des warnings pour un affichage plus propre
warnings.filterwarnings('ignore')

# Import conditionnel de Hopsworks
try:
    import hopsworks
    from hsfs.feature_store import FeatureStore
    HOPSWORKS_AVAILABLE = True
    FeatureStoreType = FeatureStore
except ImportError:
    print("⚠️ Hopsworks non disponible - mode local activé")
    HOPSWORKS_AVAILABLE = False
    FeatureStoreType = Any  # Use Any as fallback type

class AQIFeaturePipeline:
    """
    📊 Pipeline de collecte et traitement des features AQI
    
    Responsabilités:
    - Récupération données API AQICN.org
    - Feature engineering (temporelles, dérivées, etc.)
    - Validation et nettoyage des données
    - Sauvegarde Feature Store ou locale
    """
    
    def __init__(self, api_key: str, city: str = "paris"):
        """
        Initialise le pipeline de features
        
        Args:
            api_key: Clé API AQICN.org
            city: Ville pour la collecte de données
        """
        self.api_key = api_key
        self.city = city.lower()
        self.base_url = "https://api.aqicn.org/feed"
        
        # Statistiques de session
        self.stats = {
            'records_processed': 0,
            'api_calls': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        print(f"🚀 Pipeline initialisé pour {city} avec API key: {'***' + api_key[-4:] if len(api_key) > 4 else 'demo'}")

    # ===============================
    # RÉCUPÉRATION DONNÉES API
    # ===============================
    
    def fetch_aqi_data(self) -> Optional[Dict]:
        """
        Récupère les données AQI depuis l'API AQICN.org
        
        Returns:
            Dict contenant les données AQI ou None si erreur
        """
        url = f"{self.base_url}/{self.city}/?token={self.api_key}"
        self.stats['api_calls'] += 1
        
        try:
            print(f"🔗 Appel API: {self.city}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'ok':
                aqi_value = data['data'].get('aqi', 'N/A')
                print(f"✅ Données récupérées - AQI {self.city}: {aqi_value}")
                return data['data']
            else:
                print(f"⚠️ Erreur API: {data}")
                self.stats['errors'] += 1
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout API pour {self.city} (>30s)")
            self.stats['errors'] += 1
            return None
        except requests.exceptions.RequestException as e:
            print(f"🌐 Erreur réseau: {e}")
            self.stats['errors'] += 1
            return None
        except Exception as e:
            print(f"❌ Erreur inattendue: {e}")
            self.stats['errors'] += 1
            return None

    # ===============================
    # TRAITEMENT ET FEATURE ENGINEERING
    # ===============================
    
    def process_raw_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Traite les données brutes et génère les features
        
        Args:
            raw_data: Données brutes de l'API
            
        Returns:
            DataFrame avec toutes les features
        """
        if not raw_data:
            return pd.DataFrame()
        
        timestamp = datetime.now()
        
        # Extraction des données principales
        aqi_value = raw_data.get('aqi', 0)
        iaqi = raw_data.get('iaqi', {})
        
        # Features de base
        base_features = {
            'timestamp': timestamp,
            'city': self.city,
            'aqi': float(aqi_value) if aqi_value else 0.0,
            'pm25': self._extract_pollutant_value(iaqi, 'pm25'),
            'pm10': self._extract_pollutant_value(iaqi, 'pm10'),
            'o3': self._extract_pollutant_value(iaqi, 'o3'),
            'no2': self._extract_pollutant_value(iaqi, 'no2'),
            'so2': self._extract_pollutant_value(iaqi, 'so2'),
            'co': self._extract_pollutant_value(iaqi, 'co'),
            'temp': self._extract_pollutant_value(iaqi, 't', default=20.0),
            'humidity': self._extract_pollutant_value(iaqi, 'h', default=50.0),
            'pressure': self._extract_pollutant_value(iaqi, 'p', default=1013.0),
            'wind_speed': self._extract_pollutant_value(iaqi, 'w', default=0.0),
        }
        
        # Features temporelles
        temporal_features = self._generate_temporal_features(timestamp)
        
        # Features dérivées
        derived_features = self._generate_derived_features(base_features)
        
        # Combinaison de toutes les features
        all_features = {**base_features, **temporal_features, **derived_features}
        
        # Nettoyage et validation
        all_features = self._clean_features(all_features)
        
        df = pd.DataFrame([all_features])
        self.stats['records_processed'] += 1
        
        print(f"📊 Features générées: {len(all_features)} colonnes")
        self._print_feature_summary(all_features)
        
        return df
    
    def _extract_pollutant_value(self, iaqi: Dict, pollutant: str, default: float = 0.0) -> float:
        """Extrait la valeur d'un polluant de manière sécurisée"""
        try:
            value = iaqi.get(pollutant, {}).get('v', default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _generate_temporal_features(self, timestamp: datetime) -> Dict:
        """Génère les features temporelles"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month,
            'year': timestamp.year,
            'is_weekend': timestamp.weekday() >= 5,
            'season': self._get_season(timestamp.month),
            'is_rush_hour': timestamp.hour in [7, 8, 9, 17, 18, 19],
            'is_night': timestamp.hour in [22, 23, 0, 1, 2, 3, 4, 5],
            'day_of_year': timestamp.timetuple().tm_yday,
            'week_of_year': timestamp.isocalendar()[1]
        }
    
    def _generate_derived_features(self, base_features: Dict) -> Dict:
        """Génère les features dérivées"""
        pm25 = base_features['pm25']
        pm10 = base_features['pm10']
        temp = base_features['temp']
        humidity = base_features['humidity']
        
        return {
            'pm_ratio': pm25 / max(pm10, 1.0),  # Éviter division par zéro
            'pollution_score': (pm25 * 0.4 + pm10 * 0.3 + 
                              base_features['o3'] * 0.2 + base_features['no2'] * 0.1),
            'temp_humidity_index': temp * humidity / 100.0,
            'air_quality_category': self._categorize_aqi(base_features['aqi']),
            'comfort_index': self._calculate_comfort_index(temp, humidity),
            'pollutant_diversity': self._calculate_pollutant_diversity(base_features),
            'wind_pollution_factor': base_features['wind_speed'] * base_features['pm25'],
            'atmospheric_pressure_normalized': (base_features['pressure'] - 1013) / 100
        }
    
    def _clean_features(self, features: Dict) -> Dict:
        """Nettoie et valide les features"""
        cleaned = features.copy()
        
        # Nettoyage des valeurs négatives pour les polluants
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        for col in pollutant_cols:
            if col in cleaned:
                cleaned[col] = max(0.0, cleaned[col])
        
        # Validation des plages de valeurs
        if 'humidity' in cleaned:
            cleaned['humidity'] = max(0.0, min(100.0, cleaned['humidity']))
        
        if 'wind_speed' in cleaned:
            cleaned['wind_speed'] = max(0.0, cleaned['wind_speed'])
        
        if 'temp' in cleaned:
            cleaned['temp'] = max(-50.0, min(60.0, cleaned['temp']))
        
        # Remplacement des NaN/inf
        for key, value in cleaned.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    cleaned[key] = 0.0
        
        return cleaned
    
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
    
    def _categorize_aqi(self, aqi: float) -> int:
        """Catégorise l'AQI selon les standards internationaux"""
        if aqi <= 50:
            return 0  # Bon
        elif aqi <= 100:
            return 1  # Modéré
        elif aqi <= 150:
            return 2  # Malsain pour groupes sensibles
        elif aqi <= 200:
            return 3  # Malsain
        elif aqi <= 300:
            return 4  # Très malsain
        else:
            return 5  # Dangereux
    
    def _calculate_comfort_index(self, temp: float, humidity: float) -> float:
        """Calcule un index de confort basé sur température et humidité"""
        # Index simple basé sur les conditions optimales (20°C, 50% humidité)
        temp_factor = 1 - abs(temp - 20) / 40
        humidity_factor = 1 - abs(humidity - 50) / 50
        return max(0, (temp_factor + humidity_factor) / 2)
    
    def _calculate_pollutant_diversity(self, features: Dict) -> float:
        """Calcule la diversité des polluants (nombre de polluants > 0)"""
        pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        active_pollutants = sum(1 for p in pollutants if features.get(p, 0) > 0)
        return active_pollutants / len(pollutants)
    
    def _print_feature_summary(self, features: Dict):
        """Affiche un résumé des features générées"""
        aqi = features.get('aqi', 0)
        pm25 = features.get('pm25', 0)
        temp = features.get('temp', 0)
        print(f"   └── AQI: {aqi:.1f}, PM2.5: {pm25:.1f}, Temp: {temp:.1f}°C")

    # ===============================
    # GÉNÉRATION DONNÉES HISTORIQUES
    # ===============================
    
    def backfill_historical_data(self, days: int = 30):
        """
        Génère des données historiques réalistes pour l'entraînement
        
        Args:
            days: Nombre de jours à générer
        """
        print(f"🔄 Génération de {days} jours de données historiques pour {self.city}...")
        
        # Connecter au feature store
        fs = self.connect_to_hopsworks()
        
        historical_data = []
        total_hours = days * 24
        
        # Patterns spécifiques par ville
        city_patterns = self._get_city_patterns()
        
        for i in range(total_hours):
            # Date dans le passé
            base_date = datetime.now() - timedelta(hours=total_hours - i)
            
            # Génération de données réalistes avec patterns
            features = self._generate_historical_record(base_date, city_patterns)
            historical_data.append(features)
            
            # Progress indicator
            if i % (total_hours // 10) == 0:
                progress = (i / total_hours) * 100
                print(f"   📈 Progression: {progress:.0f}%")
        
        # Conversion en DataFrame
        df_historical = pd.DataFrame(historical_data)
        
        # Sauvegarde
        if fs:
            self.save_to_feature_store(df_historical, fs)
        else:
            self._save_to_local_file(df_historical, f"historical_features_{self.city}_{days}days.csv")
        
        self.stats['records_processed'] += len(historical_data)
        print(f"✅ {len(historical_data)} enregistrements historiques générés")
    
    def _get_city_patterns(self) -> Dict:
        """Retourne les patterns spécifiques par ville"""
        patterns = {
            'paris': {'base_aqi': 85, 'temp_base': 12, 'humidity_base': 70, 'pollution_factor': 1.1},
            'barcelona': {'base_aqi': 75, 'temp_base': 18, 'humidity_base': 65, 'pollution_factor': 0.9},
            'london': {'base_aqi': 65, 'temp_base': 10, 'humidity_base': 80, 'pollution_factor': 0.8},
            'madrid': {'base_aqi': 90, 'temp_base': 15, 'humidity_base': 60, 'pollution_factor': 1.2},
            'berlin': {'base_aqi': 70, 'temp_base': 8, 'humidity_base': 75, 'pollution_factor': 0.85},
            'rome': {'base_aqi': 80, 'temp_base': 16, 'humidity_base': 68, 'pollution_factor': 1.0}
        }
        
        return patterns.get(self.city, patterns['paris'])
    
    def _generate_historical_record(self, timestamp: datetime, city_patterns: Dict) -> Dict:
        """Génère un enregistrement historique réaliste"""
        # Facteurs temporels
        hour_factor = np.sin(2 * np.pi * timestamp.hour / 24) * 0.3 + 1
        day_factor = 1.2 if timestamp.weekday() < 5 else 0.8  # Plus pollué en semaine
        seasonal_factor = self._get_seasonal_factor(timestamp.month)
        weekend_factor = 0.8 if timestamp.weekday() >= 5 else 1.0
        
        # AQI de base avec variations
        base_aqi = city_patterns['base_aqi']
        aqi = base_aqi * hour_factor * day_factor * seasonal_factor * weekend_factor
        aqi += np.random.normal(0, 15)  # Bruit aléatoire
        aqi = max(10, min(300, aqi))
        
        # Données météo réalistes
        temp_base = city_patterns['temp_base']
        temp = temp_base + 10 * np.sin(2 * np.pi * (timestamp.month - 1) / 12)  # Variation saisonnière
        temp += np.random.normal(0, 3)  # Bruit
        
        humidity = city_patterns['humidity_base'] + np.random.normal(0, 15)
        humidity = max(10, min(95, humidity))
        
        # Polluants basés sur l'AQI
        pollution_factor = city_patterns['pollution_factor']
        
        features = {
            'timestamp': timestamp,
            'city': self.city,
            'aqi': aqi,
            'pm25': max(0, aqi * 0.6 * pollution_factor + np.random.normal(0, 8)),
            'pm10': max(0, aqi * 0.8 * pollution_factor + np.random.normal(0, 12)),
            'o3': max(0, aqi * 0.4 + np.random.normal(0, 6)),
            'no2': max(0, aqi * 0.3 + np.random.normal(0, 5)),
            'so2': max(0, aqi * 0.2 + np.random.normal(0, 3)),
            'co': max(0, aqi * 0.1 + np.random.normal(0, 2)),
            'temp': temp,
            'humidity': humidity,
            'pressure': np.random.normal(1013, 8),
            'wind_speed': max(0, np.random.normal(4, 2))
        }
        
        # Ajout des features temporelles et dérivées
        temporal_features = self._generate_temporal_features(timestamp)
        derived_features = self._generate_derived_features(features)
        
        all_features = {**features, **temporal_features, **derived_features}
        return self._clean_features(all_features)
    
    def _get_seasonal_factor(self, month: int) -> float:
        """Retourne un facteur saisonnier pour la pollution"""
        # Plus pollué en hiver (chauffage, inversions thermiques)
        if month in [11, 12, 1, 2]:
            return 1.2
        elif month in [6, 7, 8]:
            return 0.9  # Moins pollué en été
        else:
            return 1.0

    # ===============================
    # SAUVEGARDE DONNÉES
    # ===============================
    
    def connect_to_hopsworks(self) -> Optional[Any]:
        """Se connecte à Hopsworks et retourne le feature store"""
        if not HOPSWORKS_AVAILABLE:
            print("⚠️ Hopsworks non disponible - mode local")
            return None
        
        try:
            project = hopsworks.login(
                api_key_value=os.getenv('HOPSWORKS_API_KEY'),
                project="aqi_prediction"
            )
            print("✅ Connexion à Hopsworks réussie")
            return project.get_feature_store()
        except Exception as e:
            print(f"❌ Erreur de connexion à Hopsworks: {e}")
            print("🔄 Basculement en mode local")
            return None
    
    def save_to_feature_store(self, df: pd.DataFrame, fs: Optional[Any]):
        """Sauvegarde les features dans le feature store ou localement"""
        if fs is None:
            # Sauvegarde locale
            filename = f"features_{self.city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self._save_to_local_file(df, filename)
            return
        
        try:
            # Sauvegarde dans Hopsworks
            aqi_fg = fs.get_or_create_feature_group(
                name="aqi_features",
                version=1,
                description="Features AQI pour prédiction qualité de l'air",
                primary_key=["timestamp", "city"],
                event_time="timestamp"
            )
            
            aqi_fg.insert(df)
            print(f"✅ {len(df)} enregistrements sauvegardés dans Hopsworks")
            
        except Exception as e:
            print(f"❌ Erreur Hopsworks: {e}")
            # Fallback vers sauvegarde locale
            filename = f"backup_features_{self.city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self._save_to_local_file(df, filename)
    
    def _save_to_local_file(self, df: pd.DataFrame, filename: str):
        """Sauvegarde locale en CSV"""
        try:
            df.to_csv(filename, index=False)
            print(f"💾 Sauvegarde locale: {filename}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde locale: {e}")

    # ===============================
    # VALIDATION ET QUALITÉ
    # ===============================
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """Valide la qualité des données"""
        issues = []
        
        if df.empty:
            issues.append("DataFrame vide")
            return False, issues
        
        # Vérifications de base
        required_cols = ['aqi', 'pm25', 'pm10', 'timestamp', 'city']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Colonnes manquantes: {missing_cols}")
        
        # Validation des valeurs
        for _, row in df.iterrows():
            aqi_value = row.get('aqi', 0)
            if not (0 <= aqi_value <= 500):
                issues.append(f"AQI hors limites: {aqi_value}")
            
            pm25 = row.get('pm25', 0)
            pm10 = row.get('pm10', 0)
            if pm25 > pm10 and pm10 > 0:
                issues.append(f"PM2.5 > PM10: {pm25} > {pm10}")
            
            if row.get('humidity', 50) > 100:
                issues.append(f"Humidité > 100%: {row.get('humidity')}")
        
        return len(issues) == 0, issues

    # ===============================
    # EXÉCUTION PIPELINE
    # ===============================
    
    def run_feature_pipeline(self) -> bool:
        """Exécute le pipeline de features complet"""
        print("🚀 Démarrage du pipeline de features AQI...")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌍 Ville: {self.city}")
        
        try:
            # Récupération des données en temps réel
            raw_data = self.fetch_aqi_data()
            if not raw_data:
                print("❌ Impossible de récupérer les données, arrêt du pipeline")
                return False
            
            # Traitement des données
            df = self.process_raw_data(raw_data)
            if df.empty:
                print("❌ Aucune donnée à traiter")
                return False
            
            # Validation de la qualité
            is_valid, issues = self.validate_data_quality(df)
            if not is_valid:
                print(f"⚠️ Problèmes de qualité détectés: {issues}")
            
            # Connexion et sauvegarde
            fs = self.connect_to_hopsworks()
            self.save_to_feature_store(df, fs)
            
            # Statistiques finales
            self._print_final_stats()
            
            print("✅ Pipeline de features terminé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur dans le pipeline: {e}")
            self.stats['errors'] += 1
            return False
    
    def _print_final_stats(self):
        """Affiche les statistiques finales"""
        duration = datetime.now() - self.stats['start_time']
        print(f"\n📊 Statistiques de session:")
        print(f"   ⏱️ Durée: {duration}")
        print(f"   📝 Enregistrements traités: {self.stats['records_processed']}")
        print(f"   🔗 Appels API: {self.stats['api_calls']}")
        print(f"   ❌ Erreurs: {self.stats['errors']}")

# ===============================
# CLI ET FONCTION PRINCIPALE
# ===============================

def main():
    """Fonction principale avec gestion des arguments"""
    parser = argparse.ArgumentParser(
        description='📊 Pipeline de collecte de features AQI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python feature_pipeline.py                              # Collecte Paris temps réel
  python feature_pipeline.py --city barcelona             # Collecte Barcelona
  python feature_pipeline.py --backfill --days 30         # 30 jours historiques
  python feature_pipeline.py --city london --backfill     # Backfill London
        """
    )
    
    parser.add_argument('--city', type=str, default='paris',
                        help='Ville pour collecter les données (défaut: paris)')
    parser.add_argument('--backfill', action='store_true',
                        help='Mode backfill pour données historiques')
    parser.add_argument('--days', type=int, default=30,
                        help='Nombre de jours pour le backfill (défaut: 30)')
    
    args = parser.parse_args()
    
    # Configuration
    api_key = os.getenv('AQICN_API_KEY', 'demo')
    city = args.city.lower()
    
    print(f"🔧 Configuration:")
    print(f"   Ville: {city}")
    print(f"   Mode: {'Backfill' if args.backfill else 'Temps réel'}")
    if args.backfill:
        print(f"   Jours: {args.days}")
    print(f"   API Key: {'***' + api_key[-4:] if len(api_key) > 4 else 'demo'}")
    
    # Création et exécution du pipeline
    pipeline = AQIFeaturePipeline(api_key, city)
    
    try:
        if args.backfill:
            pipeline.backfill_historical_data(args.days)
            success = True
        else:
            success = pipeline.run_feature_pipeline()
        
        # Code de sortie pour CI/CD
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()