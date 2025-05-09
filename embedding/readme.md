Cette partie contient des scripts et notebooks pour explorer les embeddings.

## Librairies python requises
- `ollama` : pour accéder aux modèles locaux, notamment le modèle embedding `nomic-embed-text`
- `numpy` : pour les calculs
- `matplotlib.pyplot` : pour les visualisations
- `pandas` : pour manipuler les données tabulaires
- `PIL` : pour charger et afficher des images

## Fichiers requis
- `responses_with_final_success.csv` : Ce fichier CSV contient les vecteurs déjà calculés, pour éviter la perte de temps, il faut le télécharger pour avoir accès aux données
- `images` : Le dossier `images` contient des images qui sont utilisées dans les démonstrations

## Principales fonctionnalités
- Extraction d'embeddings à partir de textes avec des modèles Ollama
- Calcul de similarité entre concepts (distance euclidienne et angulaire)
- Visualisation des relations sémantiques entre mots
- Analyse comparative des réponses de modèles VLM
- Évaluation de la qualité des prédictions par méthodes d'embedding

## Utilisation
Le code principal est fourni sous forme d'un notebook Jupyter (`embedding_test.ipynb`) qui contient tout le code. `embedding.ipynb` est la version finale qui contient toutes les parties importantes pour la démonstration, et qui est plus facile à lire et à exécuter.

Après avoir téléchargé les fichiers et bibliothèques pertinents, il suffit simplement de lancer tout le fichier `embedding.ipynb`. Il existe quelques lignes qui sont prêtes à être re-exécutées plusieurs fois pour voir des exemples sur plusieurs images ou prompts, comme les parties utilisant `compare_model_responses()` et `full_guesser()`.

## Fonctions principales
- `get_vector(model, text)` : obtient l'embedding d'un texte depuis un modèle
- `get_difference(v1, v2)` : calcule la distance euclidienne entre deux vecteurs
- `get_angular_difference(v1, v2)` : calcule la distance angulaire (cosinus) entre deux vecteurs
- `compress_to_2d(vector)` : compresse un vecteur en 2D pour visualisation

## Fonctions de démonstrations
- `plot_word_differences()` : visualise les différences entre listes de mots
- `analyze_word_and_wordlist()` : analyse la proximité sémantique entre un mot et une liste
- `compare_model_responses()` : comparaison des réponses des différents modèles sur le même prompt et image
- `full_guesser()` : démonstration de l'évaluation finale de la réponse d'un prompt

**Note :** Pour exécuter les analyses, vous aurez besoin d'un modèle d'embedding installé via Ollama, de préférence `nomic-embed-text`.