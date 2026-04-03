# Projet 7 – POC RAG Puls-Events

## Objectif

Développer un Proof of Concept (POC) d’un système RAG (Retrieval-Augmented Generation)
permettant de répondre à des questions sur des événements culturels à partir de données
issues de l’API OpenAgenda.

Le système utilisera :

* LangChain
* FAISS
* Mistral
* FastAPI
* Données OpenAgenda filtrées par zone géographique et période

## Environnement de développement

Python >= 3.8
Environnement virtuel : `.venv`

Création de l’environnement :

```bash
python -m venv .venv
source .venv/bin/activate
```

Installation des dépendances :

```bash
pip install -r requirements.txt
```

## Test de l’environnement

```bash
python test_imports.py
```

Résultat attendu :

```text
Imports OK
```

## Structure du projet

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── fetch_openagenda.py
│   ├── check_dataset.py
│   ├── build_vector_index_chunks.py
│   └── test_vector_search.py
│
├── tests/
│   ├── test_clean_events.py
│   ├── test_chatbot_rag.py
│   └── test_api.py
│
├── vectorstore/
│   └── faiss_index_chunks/
│       ├── index.faiss
│       └── index.pkl
│
├── api.py
├── api_test.py
├── requirements.txt
├── README.md
├── test_imports.py
├── .env
└── .gitignore
```

Description :

* data/raw : données brutes OpenAgenda
* data/processed : dataset nettoyé prêt pour indexation
* scripts : scripts de récupération et nettoyage
* tests : tests unitaires
* requirements.txt : dépendances Python
* test_imports.py : vérification de l’environnement

## Étape 2 – Récupération et nettoyage des données OpenAgenda

Script :

`scripts/fetch_openagenda.py`

Ce script :

* appelle l’API OpenAgenda
* récupère les événements
* filtre par période (± 1 an)
* filtre par localisation
* nettoie les champs utiles
* génère un dataset CSV

Exécution :

```bash
python scripts/fetch_openagenda.py
```

Fichiers générés :

```text
data/raw/openagenda_lille.json
data/processed/events_lille_clean.csv
```

Vérification du dataset :

```bash
python scripts/check_dataset.py
```

Tests unitaires :

```bash
pytest
```

---

## Étape 3 – Indexation vectorielle avec FAISS

Script :


`scripts/build_vector_index_chunks.py`


Pipeline :

1. Chargement du dataset nettoyé
2. Création d’un document par événement
3. Découpage en chunks avec RecursiveCharacterTextSplitter
4. Génération des embeddings avec sentence-transformers
5. Création d’un index FAISS
6. Sauvegarde de l’index

Exécution :

```bash
python scripts/build_vector_index_chunks.py
```

Sortie :

```text
vectorstore/faiss_index_chunks/
index.faiss
index.pkl
```

### Vérification de l’indexation

Le script vérifie que :

- tous les événements du dataset sont convertis en documents
- tous les événements sont présents dans les chunks
- aucun événement n’est perdu lors de l’indexation

Exemple :

Événements dans le dataset : 98
Documents créés : 98
Chunks créés : 101
Tous les événements ont bien été pris en compte.


### Choix de l’index FAISS

Un index `IndexFlatL2` est utilisé.

Ce choix est adapté à un POC avec un petit volume de données,
et permet une recherche sémantique rapide et exacte.

---

### Test de recherche sémantique

Script :

`scripts/test_vector_search.py`

Permet de :

- charger l’index FAISS
- envoyer une requête texte
- récupérer les événements les plus proches

Exécution :

```bash
python scripts/test_vector_search.py
```

Exemple de requête :

```text
concert musique lille
```

Résultat attendu :

- événements pertinents retournés
- métadonnées conservées
- recherche fonctionnelle

---
## Étape 4 – Chatbot RAG avec Mistral

Script :

`scripts/chatbot_rag.py`

Description :

Ce script implémente un système RAG (Retrieval-Augmented Generation) permettant de :

- récupérer les événements les plus pertinents via FAISS
- construire un contexte à partir des documents retrouvés
- générer une réponse avec le modèle Mistral

Pipeline :

question utilisateur → recherche FAISS → récupération des chunks → construction du prompt → génération de la réponse avec Mistral

Exécution :

```bash
python scripts/chatbot_rag.py
```

Fonctionnement :

- le chatbot fonctionne en mode interactif
- plusieurs questions peuvent être posées sans relancer le script
- la commande quit permet de quitter le programme

Évaluation du chatbot

Des tests ont été réalisés pour évaluer la pertinence des réponses.

Questions testées :

- Quels concerts sont prévus à Lille ?
- Quels événements gratuits sont disponibles à Lille ?
- Quels événements ont lieu à l’Aéronef à Lille ?
- Quels événements ont lieu en mars à Lille ?
- Y a-t-il des spectacles musicaux à Lille ?

Résultats :

4 réponses correctes
1 réponse partiellement correcte

Analyse :

le chatbot fournit des réponses pertinentes lorsque l’information est présente dans les données
il évite les hallucinations en indiquant lorsqu’une information est absente
certaines limites proviennent du dataset (ex : absence d’information sur la gratuité)

Points de vigilance :

présence de bruit dans les résultats FAISS (documents non pertinents)
dépendance à la qualité des données
importance du chunking et des embeddings pour la pertinence des résultats

---
## Étape 5 – API REST avec FastAPI

Fichier :

`api.py`

Description :

Une API REST a été mise en place pour exposer le système RAG.
Cette API constitue la couche d’exposition du système RAG, permettant son intégration dans une application externe.

Elle permet :

- d’interroger le chatbot via HTTP
- de reconstruire l’index vectoriel

### Lancer l’API

```bash
uvicorn api:app --reload
```
Accès à la documentation Swagger :

http://127.0.0.1:8000/docs

### Endpoints

#### POST /ask

Permet de poser une question au système RAG.

Exemple :

```json
{
  "question": "Quels événements musicaux sont prévus à Lille ?"
}
```

Réponse :

```json
{
  "question": "...",
  "response": "...",
  "status": "success"
}
```

Gestion des erreurs :

question vide → erreur 400
erreur interne → erreur 500

#### POST /rebuild

Permet de reconstruire complètement l’index vectoriel FAISS.

**Requête :**

```bash
POST /rebuild
```

Réponse :

```json
{
  "status": "success",
  "message": "Base vectorielle reconstruite avec succès."
}
```

### Tests de l’API

Deux types de tests ont été réalisés :

1. Test manuel

Via Swagger :

http://127.0.0.1:8000/docs

2. Test automatisé

Script :

api_test.py

Exécution :

```bash
python api_test.py
```

### Tests unitaires

Les tests sont exécutés avec pytest :

```bash
python -m pytest
```

L’API doit être lancée avant d’exécuter les tests API.

Ils couvrent :
- nettoyage des données
- système RAG
- endpoints API

--
## Conteneurisation avec Docker 

Le projet a été conteneurisé afin de permettre une exécution simple et reproductible en local.

### Build de l’image

```bash
docker build -t rag-api .
```
### Lancement du conteneur

```bash
docker run -p 8000:8000 --env-file .env rag-api 
```

### Accès à l’API

Swagger :
http://127.0.0.1:8000/docs

Fonctionnement :
- l’API FastAPI est exposée sur le port 8000
- les endpoints /ask et /rebuild sont disponibles
- le système RAG est entièrement opérationnel dans le conteneur

--
## Variables d’environnement

Créer un fichier `.env` :


OPENAGENDA_API_KEY=xxxxxxxxxxxx


Le fichier `.env` ne doit pas être versionné.

---

## Objectif final du projet

Construire un système RAG complet avec :

- index vectoriel FAISS
- embeddings
- pipeline LangChain
- génération avec Mistral
- API REST
- tests unitaires
- métriques d’évaluation
- conteneur Docker
- rapport technique

---
## Démonstration

Le système permet de :

- poser une question via l’API
- récupérer une réponse générée à partir des données indexées
- reconstruire dynamiquement l’index vectoriel

Exemple de scénario :

1. Lancement de l’API
2. Accès à Swagger
3. Requête : "Quels événements musicaux à Lille ?"
4. Réponse générée par le système RAG
