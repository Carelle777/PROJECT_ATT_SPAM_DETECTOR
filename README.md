# Détecteur de SMS spam pour AT&T

## 1. Le but du projet
AT&T veut bloquer les SMS spam sans jamais bloquer un vrai SMS. J'ai entraîné deux modèles qui lisent un SMS et répondent : **spam** ou **pas spam** (en anglais « ham »). Ensuite je les ai comparés et j'ai gardé le plus adapté.

## 2. Les données
- Fichier : `spam.csv` (base publique « SMS Spam Collection »), 5572 SMS en anglais au départ.
- `v1` : l'étiquette (`ham` ou `spam`), `v2` : le texte du SMS.
- **Nettoyage des doublons** : 403 SMS étaient copiés en double. Je les supprime avant de séparer les jeux, sinon une copie dans le Train et une copie dans le Test donneraient une note trop belle (le modèle reconnaîtrait le SMS par cœur). Il reste **5169 SMS**.
- 4516 vrais SMS (87,4 %) et 653 spams (12,6 %) : les classes sont déséquilibrées.
- Le fichier est lu avec l'encodage `latin-1`. Certains caractères spéciaux s'affichent bizarrement (par exemple « å£ ») : c'est un défaut du fichier d'origine.

## 3. Ce que fait le notebook, étape par étape
1. **Chargement, suppression des doublons et audit** des SMS (graphique spam / ham avec les pourcentages).
2. **Séparation en 3 jeux (stratifiée)** : 60 % **Train** (3101 SMS, pour apprendre), 20 % **Validation** (1034 SMS, pour surveiller l'apprentissage) et 20 % **Test** (1034 SMS, l'examen final, jamais vu avant la fin).
3. **Augmentation des données** : pour avoir plus de spams à apprendre, je crée de nouveaux SMS en permutant quelques mots (392 spams → 784). Je le fais **seulement sur le Train**, jamais sur la Validation ni le Test.
4. **Transformer le texte en nombres** : chaque mot reçoit un numéro (Tokenizer, 10 000 mots maximum, appris sur le Train seulement pour éviter la fuite de données). Tous les SMS sont ramenés à 50 mots (padding).
5. **Modèle Dense** (le plus simple) : Embedding (16), moyenne des mots, couche Dense de 24 neurones, Dropout 0,2, sortie sigmoïde. Perte : binary cross-entropy.
6. **Modèle LSTM** (plus compliqué) : Embedding, LSTM de 32 unités, Dropout 0,3, sortie sigmoïde. Un arrêt automatique (EarlyStopping) garde les poids du meilleur tour sur la Validation (ici le tour 4).
7. **Générateur de texte** : un modèle de langage simple (pas un GAN) qui invente des SMS de spam. Je mesure sa qualité avec le modèle Dense.
8. **Mots-signaux** : je regarde quels mots font le plus monter la probabilité de spam d'un SMS normal.
9. **Conclusion** : comparaison et choix du modèle.

Pour avoir toujours les mêmes résultats, j'ai fixé la « graine » du hasard (`tf.keras.utils.set_random_seed(42)`).

## 4. Les résultats (sur les 1034 SMS de Test, dont 130 spams)

| Modèle | Accuracy | Vrais SMS bloqués par erreur (faux positifs) | Spams qui passent (faux négatifs) |
|---|---|---|---|
| **Dense (choisi)** | 98,3 % | **3** | 15 |
| LSTM | 98,0 % | 5 | 16 |

Sur la classe spam uniquement :

| Modèle | Précision | Rappel | F1-score |
|---|---|---|---|
| Dense | 0,975 | 0,885 | 0,927 |
| LSTM | 0,958 | 0,877 | 0,916 |

- **Précision** : quand le modèle dit « spam », il a raison 97,5 fois sur 100 (Dense).
- **Rappel** : sur 100 vrais spams, il en attrape 88 (Dense).
- **F1-score** : l'équilibre entre les deux.

**Mots-signaux** : « call », « free », « txt », « reply », « text », « mobile », « prize », « stop » sont les mots qui font le plus monter la probabilité de spam.

**Spams générés** : le modèle Dense reconnaît 100 % des spams générés comme des spams (contre environ 92 % des vrais spams de la Validation), et les 100 messages générés sont tous différents.

## 5. Le modèle choisi : Dense
Les deux modèles sont très proches (l'écart représente quelques SMS sur 1034). Je garde le **Dense** parce qu'il bloque moins de vrais SMS (3 contre 5), qu'il a le meilleur F1-score, qu'il est plus simple et beaucoup plus rapide à entraîner. Pour un opérateur, bloquer un vrai SMS est pire que laisser passer un spam. Le LSTM est plus compliqué sans être meilleur sur des SMS courts.

## 6. Les limites
- Le modèle laisse passer environ 1 spam sur 9 (15 sur 130) : c'est le point faible principal.
- Le jeu de Test contient seulement 130 spams : une différence de 1 ou 2 SMS entre les deux modèles n'est pas significative.
- Le modèle dit « spam » à partir de 50 % de certitude. Monter ce seuil (par exemple à 70 %) bloquerait moins de vrais SMS, mais laisserait passer plus de spams.
- Il reste 3 vrais SMS bloqués par erreur et 15 spams qui passent : le modèle n'est pas parfait.
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