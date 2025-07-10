#!/usr/bin/env python3
"""
🌍 AQI Prediction - Orchestrateur Principal
============================================

Coordonne les différents pipelines du système de prédiction AQI:
- Feature Pipeline: Collecte données temps réel
- Training Pipeline: Entraînement ML automatisé  
- Inference Pipeline: Génération prédictions
- Streamlit App: Interface web interactive

Usage:
    python main.py                    # Menu interactif
    python main.py --app             # Lancer Streamlit
    python main.py --status          # Statut système
    python main.py --collect paris   # Collecter données Paris
"""

import os
import sys
import argparse
from datetime import datetime
import subprocess
from typing import Dict, List, Optional
from dotenv import load_dotenv

# ===============================
# CONFIGURATION INITIALE
# ===============================

# Chargement des variables d'environnement
load_dotenv()

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===============================
# IMPORTS CONDITIONNELS
# ===============================

# Imports avec gestion d'erreurs gracieuse
FEATURE_PIPELINE_AVAILABLE = False
TRAINING_PIPELINE_AVAILABLE = False
INFERENCE_PIPELINE_AVAILABLE = False

try:
    from pipelines.feature_pipeline import AQIFeaturePipeline
    FEATURE_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Feature pipeline non disponible: {e}")

try:
    from pipelines.training_pipeline import AQITrainingPipeline
    TRAINING_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Training pipeline non disponible: {e}")

try:
    from pipelines.inference_pipeline import AQIInferencePipeline
    INFERENCE_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Inference pipeline non disponible: {e}")

# ===============================
# CLASSE ORCHESTRATEUR PRINCIPALE
# ===============================

class AQIOrchestrator:
    """
    🎼 Orchestrateur principal pour tous les pipelines AQI
    
    Coordonne l'exécution des différents composants:
    - Collecte de données (Feature Pipeline)
    - Entraînement ML (Training Pipeline) 
    - Prédictions (Inference Pipeline)
    - Interface web (Streamlit App)
    """
    
    def __init__(self):
        """Initialise l'orchestrateur avec la configuration"""
        self.config = self._load_configuration()
        self.components_status = self._check_components()
        
    def _load_configuration(self) -> Dict:
        """Charge la configuration depuis les variables d'environnement"""
        return {
            'api_key': os.getenv('AQICN_API_KEY', 'demo'),
            'hopsworks_key': os.getenv('HOPSWORKS_API_KEY'),
            'default_city': os.getenv('DEFAULT_CITY', 'paris'),
            'min_samples': int(os.getenv('MIN_TRAINING_SAMPLES', '100')),
            'max_model_age': int(os.getenv('MAX_MODEL_AGE_DAYS', '7'))
        }
    
    def _check_components(self) -> Dict[str, bool]:
        """Vérifie la disponibilité des composants"""
        return {
            'feature_pipeline': FEATURE_PIPELINE_AVAILABLE,
            'training_pipeline': TRAINING_PIPELINE_AVAILABLE,
            'inference_pipeline': INFERENCE_PIPELINE_AVAILABLE,
            'streamlit_available': self._check_streamlit(),
            'api_configured': self.config['api_key'] != 'demo',
            'hopsworks_configured': bool(self.config['hopsworks_key'])
        }
    
    def _check_streamlit(self) -> bool:
        """Vérifie si Streamlit est disponible"""
        try:
            import streamlit
            return True
        except ImportError:
            return False

    # ===============================
    # INTERFACE UTILISATEUR
    # ===============================
    
    def print_banner(self):
        """Affiche la bannière de l'application"""
        print("""
        🌍 AQI PREDICTION SYSTEM 🌍
        ===========================
        
        📊 Prédiction qualité de l'air avec IA
        🔄 Pipeline MLOps automatisé
        ⚡ Temps réel + Prédictions 3 jours
        
        Composants disponibles:
        """ + self._format_components_status() + """
        """)
    
    def _format_components_status(self) -> str:
        """Formate le statut des composants pour l'affichage"""
        status_lines = []
        components = {
            'feature_pipeline': '📊 Feature Pipeline',
            'training_pipeline': '🤖 Training Pipeline', 
            'inference_pipeline': '🔮 Inference Pipeline',
            'streamlit_available': '🌐 Streamlit App'
        }
        
        for key, name in components.items():
            emoji = "✅" if self.components_status[key] else "❌"
            status_lines.append(f"        {emoji} {name}")
        
        return "\n".join(status_lines)
    
    def show_status(self):
        """Affiche le statut détaillé du système"""
        print("\n📊 STATUT DU SYSTÈME")
        print("=" * 50)
        
        # Configuration
        print("🔧 Configuration:")
        print(f"  Ville par défaut: {self.config['default_city']}")
        print(f"  API AQICN: {'✅ Configurée' if self.config['api_key'] != 'demo' else '⚠️ Demo'}")
        print(f"  Hopsworks: {'✅ Configuré' if self.config['hopsworks_key'] else '❌ Non configuré'}")
        
        # Composants
        print("\n🧩 Composants:")
        for component, available in self.components_status.items():
            emoji = "✅" if available else "❌"
            print(f"  {emoji} {component.replace('_', ' ').title()}")
        
        # Test API rapide
        self._test_api_connectivity()
        
        # Fichiers projet
        self._check_project_files()
    
    def _test_api_connectivity(self):
        """Test rapide de connectivité API"""
        print("\n🔗 Test de connectivité:")
        
        if not self.components_status['api_configured']:
            print("  ⚠️ API AQICN: Clé demo (limitée)")
            return
        
        try:
            import requests
            url = f"https://api.aqicn.org/feed/{self.config['default_city']}/?token={self.config['api_key']}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    aqi = data['data'].get('aqi', 'N/A')
                    print(f"  ✅ API AQICN: OK (AQI {self.config['default_city']}: {aqi})")
                else:
                    print(f"  ⚠️ API AQICN: Réponse invalide")
            else:
                print(f"  ❌ API AQICN: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ API AQICN: {str(e)[:50]}...")
    
    def _check_project_files(self):
        """Vérifie la présence des fichiers du projet"""
        print("\n📁 Fichiers du projet:")
        files_to_check = [
            "pipelines/feature_pipeline.py",
            "pipelines/training_pipeline.py", 
            "pipelines/inference_pipeline.py",
            "app/streamlit_app.py",
            "app/utils/aqi_utils.py",
            "app/utils/plotting.py",
            "requirements.txt",
            ".env"
        ]
        
        for file_path in files_to_check:
            status = "✅" if os.path.exists(file_path) else "❌"
            print(f"  {status} {file_path}")

    # ===============================
    # EXÉCUTION DES PIPELINES
    # ===============================
    
    def run_feature_collection(self, city: Optional[str] = None, backfill_days: int = 0) -> bool:
        """Exécute la collecte de features"""
        if not self.components_status['feature_pipeline']:
            print("❌ Feature Pipeline non disponible")
            return False
        
        city = city or self.config['default_city']
        
        print(f"\n📊 COLLECTE DE FEATURES - {city.upper()}")
        print("=" * 50)
        
        try:
            pipeline = AQIFeaturePipeline(self.config['api_key'], city)
            
            if backfill_days > 0:
                print(f"🔄 Mode backfill: {backfill_days} jours")
                pipeline.backfill_historical_data(backfill_days)
                return True
            else:
                print("📡 Mode collecte temps réel")
                return pipeline.run_feature_pipeline()
                
        except Exception as e:
            print(f"❌ Erreur lors de la collecte: {e}")
            return False
    
    def run_model_training(self, model_type: str = 'auto') -> bool:
        """Exécute l'entraînement de modèle"""
        if not self.components_status['training_pipeline']:
            print("❌ Training Pipeline non disponible")
            return False
        
        print(f"\n🤖 ENTRAÎNEMENT DE MODÈLE")
        print("=" * 50)
        
        try:
            pipeline = AQITrainingPipeline()
            return pipeline.run_training_pipeline(model_type)
            
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement: {e}")
            return False
    
    def run_predictions(self, city: Optional[str] = None) -> Dict:
        """Génère des prédictions"""
        if not self.components_status['inference_pipeline']:
            print("❌ Inference Pipeline non disponible")
            return {"error": "Pipeline non disponible", "success": False}
        
        city = city or self.config['default_city']
        
        print(f"\n🔮 GÉNÉRATION DE PRÉDICTIONS - {city.upper()}")
        print("=" * 50)
        
        try:
            pipeline = AQIInferencePipeline()
            return pipeline.run_inference_pipeline(city)
            
        except Exception as e:
            print(f"❌ Erreur lors des prédictions: {e}")
            return {"error": str(e), "success": False}
    
    def run_full_pipeline(self, city: Optional[str] = None) -> bool:
        """Exécute le pipeline complet"""
        city = city or self.config['default_city']
        
        print(f"\n🚀 PIPELINE COMPLET - {city.upper()}")
        print("=" * 50)
        
        success_count = 0
        total_steps = 3
        
        # 1. Collecte de features
        print("📊 Étape 1/3: Collecte de features")
        if self.run_feature_collection(city):
            success_count += 1
            print("✅ Collecte réussie")
        else:
            print("⚠️ Collecte échouée, continuation...")
        
        # 2. Entraînement du modèle
        print("\n🤖 Étape 2/3: Entraînement du modèle")
        if self.run_model_training():
            success_count += 1
            print("✅ Entraînement réussi")
        else:
            print("⚠️ Entraînement échoué, continuation...")
        
        # 3. Génération de prédictions
        print("\n🔮 Étape 3/3: Génération de prédictions")
        result = self.run_predictions(city)
        if result.get("success"):
            success_count += 1
            print("✅ Prédictions générées")
            print("\n" + result["report"])
        else:
            print("⚠️ Prédictions échouées")
        
        # Résumé
        print(f"\n📊 RÉSUMÉ: {success_count}/{total_steps} étapes réussies")
        
        if success_count == total_steps:
            print("🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
            return True
        elif success_count > 0:
            print("⚠️ Pipeline partiellement réussi")
            return True
        else:
            print("❌ Échec complet du pipeline")
            return False

    # ===============================
    # LANCEMENT APPLICATIONS
    # ===============================
    
    def launch_streamlit_app(self):
        """Lance l'application Streamlit"""
        print("\n🌐 LANCEMENT DE L'APPLICATION WEB")
        print("=" * 50)
        
        if not self.components_status['streamlit_available']:
            print("❌ Streamlit n'est pas installé")
            print("💡 Installation: pip install streamlit")
            return False
        
        if not os.path.exists("app/streamlit_app.py"):
            print("❌ Fichier streamlit_app.py non trouvé")
            return False
        
        try:
            print("🚀 Lancement de Streamlit...")
            print("📍 Application accessible sur: http://localhost:8501")
            print("⏹️ Appuyez sur Ctrl+C pour arrêter\n")
            
            # Lancement de l'app
            subprocess.run([
                "streamlit", "run", "app/streamlit_app.py",
                "--server.port=8501",
                "--server.address=0.0.0.0"
            ])
            
        except KeyboardInterrupt:
            print("\n⏹️ Application arrêtée par l'utilisateur")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du lancement: {e}")
            return False

    # ===============================
    # MENU INTERACTIF
    # ===============================
    
    def interactive_menu(self):
        """Menu interactif pour l'utilisateur"""
        while True:
            print("\n" + "="*60)
            print("🌍 AQI PREDICTION - MENU PRINCIPAL")
            print("="*60)
            print("1. 📊 Collecter des données (temps réel)")
            print("2. 🔄 Générer données historiques (backfill)")
            print("3. 🤖 Entraîner le modèle")
            print("4. 🔮 Générer des prédictions")
            print("5. 🚀 Pipeline complet")
            print("6. 🌐 Lancer l'application web")
            print("7. 📊 Afficher le statut")
            print("8. ❌ Quitter")
            print("="*60)
            
            try:
                choice = input("Votre choix (1-8): ").strip()
                
                if choice == '1':
                    city = input(f"Ville [{self.config['default_city']}]: ").strip() or self.config['default_city']
                    self.run_feature_collection(city)
                
                elif choice == '2':
                    city = input(f"Ville [{self.config['default_city']}]: ").strip() or self.config['default_city']
                    days = input("Nombre de jours [30]: ").strip()
                    days = int(days) if days.isdigit() else 30
                    self.run_feature_collection(city, days)
                
                elif choice == '3':
                    model_type = input("Type de modèle [auto]: ").strip() or 'auto'
                    self.run_model_training(model_type)
                
                elif choice == '4':
                    city = input(f"Ville [{self.config['default_city']}]: ").strip() or self.config['default_city']
                    result = self.run_predictions(city)
                    if result.get("success"):
                        print("\n" + result["report"])
                
                elif choice == '5':
                    city = input(f"Ville [{self.config['default_city']}]: ").strip() or self.config['default_city']
                    self.run_full_pipeline(city)
                
                elif choice == '6':
                    self.launch_streamlit_app()
                
                elif choice == '7':
                    self.show_status()
                
                elif choice == '8':
                    print("👋 Au revoir!")
                    break
                
                else:
                    print("❌ Choix invalide, veuillez réessayer")
                    
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")

# ===============================
# FONCTION PRINCIPALE ET CLI
# ===============================

def main():
    """Fonction principale avec arguments CLI"""
    parser = argparse.ArgumentParser(
        description='🌍 Orchestrateur AQI Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python main.py                           # Menu interactif
  python main.py --app                     # Lancer Streamlit
  python main.py --status                  # Afficher le statut
  python main.py --collect paris           # Collecter données Paris
  python main.py --backfill --days 30      # Générer 30 jours de données
  python main.py --train                   # Entraîner le modèle
  python main.py --predict london          # Prédictions pour London
  python main.py --full-pipeline madrid    # Pipeline complet Madrid

Villes supportées:
  paris, barcelona, london, madrid, berlin, rome, amsterdam,
  brussels, vienna, zurich, lisbon, dublin, stockholm, etc.
        """
    )
    
    # Arguments CLI
    parser.add_argument('--collect', type=str, metavar='CITY',
                        help='Collecter les données pour une ville')
    parser.add_argument('--backfill', action='store_true',
                        help='Mode backfill pour données historiques')
    parser.add_argument('--days', type=int, default=30,
                        help='Nombre de jours pour le backfill (défaut: 30)')
    parser.add_argument('--train', action='store_true',
                        help='Entraîner le modèle ML')
    parser.add_argument('--predict', type=str, metavar='CITY',
                        help='Générer des prédictions pour une ville')
    parser.add_argument('--full-pipeline', type=str, metavar='CITY',
                        help='Exécuter le pipeline complet pour une ville')
    parser.add_argument('--app', action='store_true',
                        help='Lancer l\'application Streamlit')
    parser.add_argument('--status', action='store_true',
                        help='Afficher le statut du système')
    
    args = parser.parse_args()
    
    # Création de l'orchestrateur
    orchestrator = AQIOrchestrator()
    
    # Banner
    orchestrator.print_banner()
    
    # Exécution selon les arguments
    try:
        if args.collect:
            orchestrator.run_feature_collection(args.collect)
        elif args.backfill:
            city = args.collect or orchestrator.config['default_city']
            orchestrator.run_feature_collection(city, args.days)
        elif args.train:
            orchestrator.run_model_training()
        elif args.predict:
            result = orchestrator.run_predictions(args.predict)
            if result.get("success"):
                print("\n" + result["report"])
        elif args.full_pipeline:
            orchestrator.run_full_pipeline(args.full_pipeline)
        elif args.app:
            orchestrator.launch_streamlit_app()
        elif args.status:
            orchestrator.show_status()
        else:
            # Menu interactif par défaut
            orchestrator.interactive_menu()
            
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("💡 Utilisez --help pour voir l'aide")

if __name__ == "__main__":
    main()