# Utilisation de l'image Python officielle
FROM python:3.9-slim

# Métadonnées de l'image
LABEL maintainer="AQI Prediction Team"
LABEL version="1.0"
LABEL description="Application de prédiction de la qualité de l'air avec Streamlit"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Création d'un utilisateur non-root pour la sécurité
RUN groupadd -r streamlit && useradd -r -g streamlit streamlit

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code de l'application
COPY . .

# Création du répertoire .streamlit et configuration
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/config.toml

# Configuration des permissions
RUN chown -R streamlit:streamlit /app

# Changement vers l'utilisateur non-root
USER streamlit

# Exposition du port
EXPOSE 8501

# Health check pour vérifier que l'application fonctionne
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande par défaut pour lancer l'application
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]