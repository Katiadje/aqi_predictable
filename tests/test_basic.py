"""
Tests de base pour AQI Predictable
"""
import os
import sys

# Ajouter le chemin du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_project_structure():
    """Test que les fichiers principaux existent"""
    assert os.path.exists("app/streamlit_app.py")
    assert os.path.exists("pipelines/feature_pipeline.py")
    assert os.path.exists("pipelines/training_pipeline.py")
    assert os.path.exists("pipelines/inference_pipeline.py")
    assert os.path.exists("requirements.txt")
    assert os.path.exists("Dockerfile")

def test_imports():
    """Test que les modules s'importent sans erreur"""
    try:
        from app.utils.aqi_utils import AQIUtils
        from pipelines.feature_pipeline import AQIFeaturePipeline
        assert True
    except ImportError as e:
        assert False, f"Import error: {e}"

def test_aqi_utils():
    """Test des fonctions utilitaires AQI"""
    from app.utils.aqi_utils import AQIUtils
    
    # Test catégorisation AQI
    assert AQIUtils.get_aqi_category(25) == "Bon"
    assert AQIUtils.get_aqi_category(75) == "Modéré"
    assert AQIUtils.get_aqi_category(125) == "Malsain pour groupes sensibles"
    
    # Test couleurs AQI
    assert AQIUtils.get_aqi_color(25) == "#00E400"
    assert AQIUtils.get_aqi_color(75) == "#FFFF00"

def test_feature_pipeline_creation():
    """Test création pipeline de features"""
    from pipelines.feature_pipeline import AQIFeaturePipeline
    
    pipeline = AQIFeaturePipeline(api_key="demo", city="paris")
    assert pipeline.city == "paris"
    assert pipeline.api_key == "demo"
    assert pipeline.stats['records_processed'] == 0

def test_cities_available():
    """Test que les villes sont disponibles"""
    from app.utils.aqi_utils import APIHelper
    
    cities = APIHelper.get_available_cities()
    assert len(cities) > 0
    assert "paris" in cities
    assert "barcelona" in cities
    assert "london" in cities