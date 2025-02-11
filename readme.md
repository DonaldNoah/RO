# Optimisation du Déploiement des Engins de Terrassement

## 🚧 Description du Projet

Cette application Streamlit permet l'optimisation du déploiement d'engins de terrassement pour des projets de construction du Barrage de Natigal . Elle utilise des techniques de programmation linéaire pour minimiser le nombre d'engins tout en respectant les contraintes de volume et de temps.

## ✨ Fonctionnalités Principales

- Gestion dynamique du parc d'engins
- Optimisation du déploiement des équipements
- Analyse de sensibilité sur différents paramètres
- Génération de rapports Excel détaillés
- Visualisation interactive des déploiements via des diagrammes de Gantt
- Ajout et Suppression des Engins 

## 🛠️ Prérequis

- Python 3.8+
- pip (gestionnaire de packages Python)

## 🚀 Installation

1. Documentation Sur les environnement virtuel :

https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

2. Créez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## 🖥️ Lancement de l'Application

```bash
streamlit run app.py
```

## 📊 Utilisation

1. Configurez les paramètres du projet dans la barre latérale
2. Gérez votre parc d'engins (ajout/suppression)
3. Lancez l'optimisation
4. Explorez le diagramme de Gantt et téléchargez le rapport

## 🔬 Analyse de Sensibilité

L'application propose deux types d'analyse de sensibilité :
- Impact du volume total de terrassement
- Impact du temps d'attente des engins

## 📋 Personnalisation

Vous pouvez facilement :
- Ajouter de nouveaux engins
- Modifier les paramètres du projet
- Ajuster les contraintes d'optimisation 

