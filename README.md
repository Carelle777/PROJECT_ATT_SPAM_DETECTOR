# Détecteur de SMS spam pour AT&T

## 1. Le but du projet
AT&T veut bloquer les SMS spam sans jamais bloquer un vrai SMS. J'ai entraîné deux modèles qui lisent un SMS et répondent : **spam** ou **pas spam** (en anglais « ham »). Ensuite je les ai comparés et j'ai gardé le meilleur.

## 2. Les données
- Fichier : `spam.csv` (base publique « SMS Spam Collection »), 5572 SMS en anglais.
- `v1` : l'étiquette (`ham` ou `spam`), `v2` : le texte du SMS.
- Il y a beaucoup plus de vrais SMS que de spams (les classes sont déséquilibrées).
- Le fichier est lu avec l'encodage `latin-1`. Certains caractères spéciaux s'affichent bizarrement (par exemple « å£ ») : c'est un défaut du fichier d'origine.

## 3. Ce que fait le notebook, étape par étape
1. **Chargement et exploration** des SMS.
2. **Séparation 80 % / 20 %** : 80 % pour apprendre, 20 % (1115 SMS) pour tester. La séparation garde la même proportion de spams dans les deux parties (« stratifiée »).
3. **Augmentation des données** : pour avoir plus de spams à apprendre, je crée de nouveaux SMS en mélangeant l'ordre des mots. Je le fais **seulement sur la partie entraînement**, jamais sur le test, sinon les résultats seraient faux.
4. **Transformer le texte en nombres** : chaque mot reçoit un numéro (Tokenizer, 10 000 mots maximum). Tous les SMS sont ensuite ramenés à la même longueur de 50 mots (padding).
5. **Modèle Dense** (le plus simple) : un « Embedding » puis des couches Dense.
6. **Modèle LSTM** (plus compliqué) : un réseau qui lit les mots dans l'ordre, avec 32 unités.
7. **Générateur de texte** : un petit modèle qui essaie d'inventer des SMS de spam.
8. **Conclusion** : comparaison et choix du modèle.

Pour avoir toujours les mêmes résultats, j'ai fixé la « graine » du hasard avec `tf.keras.utils.set_random_seed(42)`.

## 4. Les résultats (sur 1115 SMS jamais vus)

| Modèle | Bonnes réponses (accuracy) | Vrais SMS bloqués par erreur (faux positifs) | Spams qui passent (faux négatifs) |
|---|---|---|---|
| **Dense (choisi)** | 98,7 % | **5** | 10 |
| LSTM | 96,5 % | 24 | 15 |

Sur les spams uniquement :

| Modèle | Précision | Rappel |
|---|---|---|
| Dense | 0,97 | 0,93 |
| LSTM | 0,85 | 0,90 |

- **Précision** : quand le modèle dit « spam », il a raison 97 fois sur 100 (Dense).
- **Rappel** : sur 100 vrais spams, il en attrape 93 (Dense).

## 5. Le modèle choisi : Dense
Le modèle **Dense** est meilleur partout, et surtout il bloque beaucoup moins de vrais SMS (5 contre 24). Pour un opérateur, bloquer un vrai SMS est pire que laisser passer un spam. Le LSTM est plus compliqué mais pas meilleur : il apprend trop par cœur les SMS d'entraînement (c'est l'overfitting).

## 6. Les limites
- Le même jeu de SMS sert à tester et à vérifier pendant l'entraînement : les résultats sont un peu trop optimistes.
- Le modèle dit « spam » à partir de 50 % de certitude. Monter ce seuil (par exemple à 70 %) bloquerait moins de vrais SMS, mais laisserait passer plus de spams.
- Il reste 5 vrais SMS bloqués par erreur : le modèle n'est pas parfait.
- Le générateur de texte est faible : il recopie surtout des phrases de spam qu'il a déjà vues.
- Les SMS viennent d'une base publique, pas d'AT&T : il faudrait tester sur de vrais SMS avant de l'utiliser pour de vrai.

## 7. Lancer le projet chez soi
```
git clone https://github.com/Carelle777/PROJECT_ATT_SPAM_DETECTOR.git
cd PROJECT_ATT_SPAM_DETECTOR
pip install -r requirements.txt
```
Puis ouvrir `01-AT&T_spam_detector.ipynb` dans VS Code ou Jupyter et lancer toutes les cellules (Restart + Run All).

Les modèles entraînés (fichiers `.h5`) et le tokenizer (`.pkl`) ne sont **pas** sur GitHub, car ils sont trop lourds : ils se recréent en lançant le notebook.

## 8. Outils utilisés
Python, pandas, numpy, scikit-learn, TensorFlow / Keras, matplotlib, seaborn, plotly.