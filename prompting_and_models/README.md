--- Code pour la partie Prompting et modèles ---  
(Auteur: Victor Fleiser)

# Manuel d’utilisation

Ce dossier contient les scripts permettant de tester différentes formulations de prompts sur des modèles vision-langage, ainsi que d’évaluer leur robustesse face à des transformations d’images (flou, surexposition, etc.). Les résultats sont sauvegardés sous forme de fichiers CSV et peuvent être convertis en images pour une visualisation plus intuitive.

## Librairies Python requises

- `ollama` : pour utiliser les modèles localement via l’application Ollama  
- `pandas`, `matplotlib`, `numpy` : librairies classiques  
- `csv`, `PIL` : pour gérer les fichiers CSV ainsi que les images

---

## 1. Partie : Exploration de Prompts

- `main.py`  
  Permet de tester une liste de prompts sur une liste de modèles et d’images.  
  - Les prompts sont stockés dans `ressources/all_prompts.txt`  
  - Les autres paramètres sont modifiables dans le fichier `main.py`  
  - Les réponses des modèles, ainsi que les métadonnées (prompt, modèle, temps d'exécution, etc.), sont enregistrées dans `output/responses.csv`

- `create_CSVs_from_responses.py`  
  Génère plusieurs fichiers CSV à partir de `output/responses.csv` :
  - `output/accuracy_matrix.csv`  
    - Lignes : prompts  
    - Colonnes : modèles  
    - Contenu : nombre d’images correctement classifiées pour chaque couple prompt/modèle.
  - Pour chaque image : `output/image_matrices/`  
    - Lignes : prompts  
    - Colonnes : modèles  
    - Contenu : `True` si la prédiction est correcte, `False` sinon.
  - Pour chaque modèle : `output/model_matrices/`  
    - Lignes : prompts  
    - Colonnes : images  
    - Contenu : `True` ou `False`.
  - Pour chaque prompt : `output/prompt_matrices/`  
    - Lignes : modèles  
    - Colonnes : images  
    - Contenu : `True` ou `False`.

- `csv_to_image.py`  
  Convertit les fichiers CSV générés en images PNG pour une visualisation plus facile.

---

## 2. Partie : Transformations d’Images

- `transform_images.py`  
  Script permettant de générer des variantes des images originales : floutées, occluses, surexposées, etc. Certaines transformations utilisées dans le projet ont été faites manuellement (par exemple le rognage).

- `main_2.py`  
  Permet de tester un prompt unique sur une liste de modèles et de dossiers d’images (avec différentes variantes).  
  - Les résultats sont enregistrés dans `output/responses_2.csv`

- `create_CSVs_from_responses_2.py`  
  Génère plusieurs fichiers CSV à partir de `output/responses_2.csv` :
  - `output/accuracy_matrix_2.csv`  
    - Lignes : transformations  
    - Colonnes : modèles  
    - Contenu : nombre d’images correctement classifiées par transformation/modèle.
  - Pour chaque image : `output/image_matrices_2/`  
    - Lignes : transformations  
    - Colonnes : modèles  
    - Contenu : `True` ou `False`.
  - Pour chaque modèle : `output/model_matrices_2/`  
    - Lignes : transformations  
    - Colonnes : images  
    - Contenu : `True` ou `False`.
  - Pour chaque transformation : `output/transformation_matrices_2/`  
    - Lignes : modèles  
    - Colonnes : images  
    - Contenu : `True` ou `False`.

- `csv_to_image_2.py`  
  Convertit les fichiers CSV générés en images PNG pour une visualisation plus facile.

---

## ⚠️ Important

- Pour réinitialiser tous les résultats, **supprimez manuellement le contenu du dossier `output/`**.
- Si vous souhaitez modifier ou compléter les réponses enregistrées, vous pouvez directement éditer `output/responses.csv` (ou `output/responses_2.csv` pour les transformations), puis relancer :
  - `create_CSVs_from_responses.py` et `csv_to_image.py` pour la partie prompts
  - `create_CSVs_from_responses_2.py` et `csv_to_image_2.py` pour la partie transformations

---

## 📁 Dossier `output_archives`

Nous avons laissé l’ensemble des résultats obtenus durant ce projet dans le dossier `output_archives/`, incluant :
- les réponses des modèles
- les fichiers CSV générés
- les visualisations associées
