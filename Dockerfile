# Utilisation de l'image Python
FROM python:3.12

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt depuis le dossier backend dans le conteneur
COPY backend/requirements.txt /app/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source depuis backend/ dans le conteneur
COPY backend/ /app/

# Commande pour démarrer l'application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]