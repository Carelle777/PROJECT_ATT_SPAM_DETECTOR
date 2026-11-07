# Détecteur de SMS spam pour AT&T

## 1. Le but du projet
AT&T veut bloquer les SMS spam sans jamais bloquer un vrai SMS. J'ai entraîné deux modèles qui lisent un SMS et répondent : **spam** ou **pas spam** (en anglais « ham »). Ensuite je les ai comparés et j'ai gardé le plus adapté.

## 2. Les données
- Fichier : `spam.csv` (base publique « SMS Spam Collection »), 5572 SMS en anglais.
- `v1` : l'étiquette (`ham` ou `spam`), `v2` : le texte du SMS.
- 4825 vrais SMS (86,6 %) et 747 spams (13,4 %) : les classes sont déséquilibrées.
- Le fichier est lu avec l'encodage `latin-1`. Certains caractères spéciaux s'affichent bizarrement (par exemple « å£ ») : c'est un défaut du fichier d'origine.

## 3. Ce que fait le notebook, étape par étape
1. **Chargement et audit** des SMS (graphique spam / ham avec les pourcentages).
2. **Séparation en 3 jeux (stratifiée)** : 60 % **Train** (pour apprendre), 20 % **Validation** (1114 SMS, pour surveiller l'apprentissage) et 20 % **Test** (1115 SMS, l'examen final, jamais vu avant la fin).
3. **Augmentation des données** : pour avoir plus de spams à apprendre, je crée de nouveaux SMS en permutant quelques mots (448 spams → 896). Je le fais **seulement sur le Train**, jamais sur la Validation ni le Test.
4. **Transformer le texte en nombres** : chaque mot reçoit un numéro (Tokenizer, 10 000 mots maximum, appris sur le Train seulement pour éviter la fuite de données). Tous les SMS sont ramenés à 50 mots (padding).
5. **Modèle Dense** (le plus simple) : Embedding (16), moyenne des mots, couche Dense de 24 neurones, Dropout 0,2, sortie sigmoïde. Perte : binary cross-entropy.
6. **Modèle LSTM** (plus compliqué) : Embedding, LSTM de 32 unités, Dropout 0,3, sortie sigmoïde. Un arrêt automatique (EarlyStopping) garde les poids du meilleur tour sur la Validation.
7. **Générateur de texte** : un modèle de langage simple (pas un GAN) qui invente des SMS de spam. Je mesure sa qualité avec le modèle Dense.
8. **Mots-signaux** : je regarde quels mots font le plus monter la probabilité de spam d'un SMS normal.
9. **Conclusion** : comparaison et choix du modèle.

Pour avoir toujours les mêmes résultats, j'ai fixé la « graine » du hasard (`tf.keras.utils.set_random_seed(42)`).

## 4. Les résultats (sur les 1115 SMS de Test)

| Modèle | Accuracy | Vrais SMS bloqués par erreur (faux positifs) | Spams qui passent (faux négatifs) |
|---|---|---|---|
| **Dense (choisi)** | 99,2 % | **2** | 7 |
| LSTM | 99,1 % | 4 | 6 |

Sur la classe spam uniquement :

| Modèle | Précision | Rappel | F1-score |
|---|---|---|---|
| Dense | 0,986 | 0,953 | 0,969 |
| LSTM | 0,973 | 0,960 | 0,966 |

- **Précision** : quand le modèle dit « spam », il a raison 98,6 fois sur 100 (Dense).
- **Rappel** : sur 100 vrais spams, il en attrape 95 (Dense).
- **F1-score** : l'équilibre entre les deux.

**Mots-signaux** : « call », « txt », « text », « stop », « free », « claim », « won », « 150p », « ringtone » sont les mots qui font le plus monter la probabilité de spam.

**Spams générés** : le modèle Dense reconnaît environ 99 % des spams générés comme des spams (contre environ 88 % des vrais spams de la Validation), et les 100 messages générés sont tous différents.

## 5. Le modèle choisi : Dense
Les deux modèles sont très proches (l'écart représente quelques SMS sur 1115). Je garde le **Dense** parce qu'il bloque moins de vrais SMS (2 contre 4), qu'il est plus simple et beaucoup plus rapide à entraîner. Pour un opérateur, bloquer un vrai SMS est pire que laisser passer un spam. Le LSTM est plus compliqué sans être meilleur sur des SMS courts.

## 6. Les limites
- Le jeu de Test contient seulement 150 spams : une différence de 1 ou 2 SMS entre les deux modèles n'est pas significative.
- Le modèle dit « spam » à partir de 50 % de certitude. Monter ce seuil (par exemple à 70 %) bloquerait moins de vrais SMS, mais laisserait passer plus de spams.
- Il reste 2 vrais SMS bloqués par erreur et 7 spams qui passent : le modèle n'est pas parfait.
- Le générateur de texte produit des caricatures de spam (mots répétés), pas de l'anglais naturel.
- Les SMS viennent d'une base publique, pas d'AT&T : il faudrait tester sur de vrais SMS avant de l'utiliser pour de vrai.

## 7. Lancer le projet chez soi
```
git clone https://github.com/Carelle777/PROJECT_ATT_SPAM_DETECTOR.git
cd PROJECT_ATT_SPAM_DETECTOR
pip install -r requirements.txt
```
Puis ouvrir `AT&T.ipynb` dans VS Code ou Jupyter et lancer toutes les cellules (Restart + Run All).

Les modèles entraînés (fichiers `.h5`) et le tokenizer (`.pkl`) ne sont **pas** sur GitHub, car ils sont trop lourds : ils se recréent en lançant le notebook.

## 8. Outils utilisés
Python, pandas, numpy, scikit-learn, TensorFlow / Keras, matplotlib, seaborn, plotly.