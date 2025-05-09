# Perception active pour la robotique basée sur des LLM
### Projet ANDROIDE 2024-25

**Encadrants** :
- Stéphane Doncieux
- Émiland Garrabé

**Etudiants affectés** :
- Victor FLEISER
- Thomas MARCHAND
- Tarik Ege EKEN


Présentation
=============

Ce projet s’inscrit dans une démarche exploratoire visant à intégrer les grands modèles de langage (LLM) dans des boucles de perception active pour la robotique, en particulier pour des tâches de classification d’objets par des robots manipulateurs tels que TIAGo ou des bras Franka.

L’objectif de ce projet est de concevoir et tester un module de perception active guidé par
le language, capable d’exploiter des modèles VLM pré-entraînés dans un cadre interactif.

Toutes les informations supplémentaires concernant le projet sont dans le rapport `Rapport.pdf`.

Le code est divisé en trois parties
- `prompting_and_models`: Comparaison de différents prompts et modèles de LLM et manipulation d'images.
- `uncertainty`: Création d'arbres de probabilités et calcul d'incertitude de l'identification d'objets.
- `embedding`: Utilisation d'embeddings pour évaluer la similarité entre différents mots.