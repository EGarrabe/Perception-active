--- Code pour la partie Incertitude ---
(Auteur: Thomas Marchand)

NOTE: les modèles LLM sont chargés avec une fonction dans load_ollama_models. Le code de cette partie fonctionne avec llava-v1.6, mais le ChatHandler n'est pas adapté pour d'autres modèles. Il est donc conseillé de rester sur llava 7b ou llava 13b.


====================================
1-create_probability_tree_multi.py
====================================

Crée un arbre de probabilité pour plusieurs modèles et plusieurs images.
Enregistre des graphes et les réponses générés et les place dans un fichier temporaire (OUTPUT_BASE_DIR)
Crée un csv, des images d'arbres, un graphique montrant l'evolution du résultat, un fichier txt contenant les réponses et un tableau final (image) indiquant la probabilité d'avoir le mot correct ainsi que la masse de probabilité explorée pour chaque image et modèle.

Utilisation:
NUM_PATHS_TO_FIND : nombre de chemins à générer. 10 est une bonne valeur pour tester. Il est recommendé de donner la même taille à RANKS_K.
MODEL_CONFIGS : modèles à utiliser depuis ollama.
Les images à utiliser doivent être mises dans IMAGE_FOLDER = dataset_vlm_0328_chosen.
Les mots "corrects" doivent être indiqués dans TARGET_WORDS_LIST. Chaque image contiendra une liste de mots.
Les libraries spécifiés dans le code ainsi que graphviz doivent être installés.
ollama doit être installé car le code cherche directement les modèles ollama (testé sous Windows). En cas de problème, vous pouvez onner model_path et mmproj_path manuellement.
load_ollama_models.py doit être présent.




==================
2-suffix_probas.py
==================

Crée un arbre de probabilité pour plusieurs modèles et plusieurs images après un préfix (caché).
Crée un csv et un tableau final (image) indiquant des statistiques sur le suffix de la réponse donné par le VLM.

Utilisation:
NUM_PATHS_TO_FIND: nombre de chemins à générer. 3 est une bonne valeur par défaut, toutes les réponses seront visibles. Il est recommendé de donner la même taille à RANKS_K.
TOP_K_FOR_PREFIX_SEARCH: les top k premiers tokens à regarder pour match avec notre préfixe. Si le préfixe n'est pas détecté (rare), augmenter cette valeur.
MODEL_CONFIGS : modèles à utiliser depuis ollama.
Les images à utiliser doivent être mises dans IMAGE_FOLDER = dataset_vlm_0328_chosen.
Les mots "corrects" doivent être indiqués dans TARGET_WORDS_LIST. Chaque image contiendra une liste de mots.
Les libraries spécifiés dans le code doivent être installés.
ollama doit être installé car le code cherche directement les modèles ollama (testé sous Windows). En cas de problème, vous pouvez onner model_path et mmproj_path manuellement.
load_ollama_models.py doit être présent.




=========================
3-probs_to_classifier.py
=========================

Crée et entraîne un classifieur depuis un csv pour classifier les réponses de VLM en se basant sur les probabilités de la réponse (suffixe).
Enregistre un fichier joblib réutilisable.

Utilisation:
CSV_FILE_PATH: fichier csv à utiliser. Il est compatible avec le csv output de 2-suffix_probas.py.
TEST_SET_SIZE: proportion des réponses à utiliser pour le split test.




===========================
4-idenfify_with_confidence
===========================

Le script 2 et 3 combinés pour obtenir une réponse du VLM de son identification de l'objet depuis plusieurs images, ainsi que l'incertitude donnée par le classifieur.
Elle simule le robot qui redemande des images jusqu'à ce qu'elle soit assez sure de son identification d'objet.
(Attention, la classification peut-être fausse, même avec le seuil Zero False Positives.)

Utilisation:
IMAGE_DIR: chemin du fichier contenant les images
IMAGE_NAMES: noms d'images à utiliser pour l'objet à identifier (pour simuler un robot prenant plusieurs images).

MODEL_NAME, MODEL_PATH, MMPROJ_PATH = modèle LLM utilisé, spécifier manuellement des modèles .gguf si ollama non installé ou ne fonctionne pas (testé sous Windows 11)
CONFIDENCE_TARGET_TYPE: type du seuil à utiliser pour définir la confiance nécessaire pour classifier comme "Sûr".
CLASSIFIER_MODEL_PATH = chemin du classifieur .joblib. Il doit être entraîné sur le même type d'image, les mêmes features et le même modèle pour obtenir une performance acceptable.