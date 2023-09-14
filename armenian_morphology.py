# -*- coding: utf-8 -*-
"""Armenian Morphology

Original file is located at
    https://colab.research.google.com/drive/1PyDTfZ8kqZDBTQ55rExri3ksCa9wGexx

I used the Universal Dependencies Armenian dataset found [here](https://github.com/UniversalDependencies/UD_Armenian-ArmTDP).

In this section, I am loading the dataset (downloaded to Google Drive) into Google Colab and unzipping it in My Drive.
"""

from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/My Drive/UD_Armenian-ArmTDP-master.zip" -d "/content/drive/My Drive/UD_Armenian-ArmTDP-master"

"""Since all the files are in ConLLU format, here I am importing pyconll to interpret the files and instantiating `path` variables for each of the datasets needed to train the machine learning model"""

!pip install pyconll

import pyconll

#loads the training dataset
train_path = '/content/drive/MyDrive/UD_Armenian-ArmTDP-master/hy_armtdp-ud-test.conllu'

#loads the development dataset
dev_path = '/content/drive/MyDrive/UD_Armenian-ArmTDP-master/hy_armtdp-ud-dev.conllu'

#loads the test dataset
test_path = '/content/drive/MyDrive/UD_Armenian-ArmTDP-master/hy_armtdp-ud-test.conllu'

"""Here I made the `load_corpus()` function. The purpose of this function is to load the dataset for preprocessing. I created two lists, one `verbs` (to compile all occurences of VERB tokens) and `infinitives`, the lemmatized versions of the said verb tokens. Each token is appended to their respective list in a fully lowercase form for ease of processing."""

import os
from transliterate import translit
import pyconll

corpus_dir = '/content/drive/MyDrive/UD_Armenian-ArmTDP-master'

def load_corpus():
    verbs = []
    infinitives = []

    train_data = pyconll.load_from_file(os.path.join(corpus_dir, 'hy_armtdp-ud-train.conllu'))
    dev_data = pyconll.load_from_file(os.path.join(corpus_dir, 'hy_armtdp-ud-dev.conllu'))
    test_data = pyconll.load_from_file(os.path.join(corpus_dir, 'hy_armtdp-ud-test.conllu'))

    for data in [train_data, dev_data, test_data]:
        for sentence in data:
            for token in sentence:
                if token.upos == 'VERB':
                    verbs.append(translit(token.form.lower(), 'hy', reversed=True))
                    infinitives.append(translit(token.lemma.lower(), 'hy', reversed=True))

    return verbs, infinitives

"""Here is where I begin the preprocessing. I start with lemmatizing the `verbs` list. Using the polyglot library (and the other libraries necessary for Polyglot to function), I created a `Text` object with 'hy' (language code for Armenian) as a parameter. Then, the lemma is instantiated as the verb's first morpheme (the `.morphemes` method takes an `Text` object and returns the morphemes for the word. The first morpheme will always be the lemma, which is why the 0th index is used for the lemma variable)"""

!pip install polyglot
!pip install pyicu
!pip install pycld2
!pip install morfessor
!polyglot download morph2.hy

from polyglot.text import Text
from transliterate import translit

def lemmatize_verbs(verbs):
    lemmas = []
    for verb in verbs:
        # Transliterate verb to Armenian script
        verb_armenian = translit(verb, 'hy', reversed=True)
        text = Text(verb_armenian, hint_language_code='hy')
        lemma = text.morphemes[0]
        lemmas.append(lemma)

    return lemmas

lemmas = lemmatize_verbs(verbs)

"""After lemmatizing the verbs, I created a `preprocess` function.

The purpose of this function is to clean and filter the lemmas from the previous function. This function removes stop words (such as tense markers and auxiliary verbs). I manually provided a list of stop words from [this github repository](https://github.com/stopwords-iso/stopwords-hy/blob/master/stopwords-hy.txt) and stripped each lemma if it happened to contain one of the stop words.

Rare words (determined by frequency of appearance within the data (i.e., `min_count`)) are also accounted for and are not included for the purpose of training the model and figuring out a consistent paradigm.
"""

from collections import Counter
from transliterate import translit

def preprocess(lemmas, min_count=5):
    stop_words = set(['այդ', 'այլ', 'այն', 'այս', 'դու', 'դուք', 'եմ', 'են', 'ենք', 'ես', 'եք', 'է', 'էի', 'էին', 'էինք', 'էիր', 'էիք', 'էր', 'ըստ', 'թ', 'ի', 'ին', 'իսկ', 'իր', 'կամ', 'համար', 'հետ', 'հետո', 'մենք', 'մեջ', 'մի', 'ն', 'նա', 'նաև', 'նրա', 'նրանք', 'որ', 'որը', 'որոնք', 'որպես', 'ու', 'ում', 'պիտի', 'վրա', 'և'])
    stop_words_latin = set([translit(word, 'hy', reversed=True) for word in stop_words])
    counter = Counter(lemmas)
    rare_words = set([word for word in counter if counter[word] < min_count])
    lemmas = [translit(lemma, 'hy', reversed=True) for lemma in lemmas if lemma not in stop_words_latin and lemma not in rare_words]

    return lemmas

"""Once the data is prepared, I use this code to train and evaluate a logistic regression model. This model classifies Armenian verbs as either infinitive or non-infinitive based on their spelling. I used `SciKit-learn` library to do this.

I first load the list of verbs by running`load_corpus()[0]`

I create a `CountVectorizer` object to transform a collection of text into a numerical feature matrix. The `lowercase=False` parameter specifies that the case of the text should not be changed.

I use `vectorizer.fit_transform(verbs)` to convert the list of verbs into a feature matrix.
Each row of the matrix corresponds to a verb, and each column corresponds to a unique word in the text. The values in the matrix represent the frequency of each word in each verb.

The labels for the data are created by checking whether each verb ends with the suffix 'ել', which is a common indicator of an infinitive verb in Armenian.

The `train_test_split` function splits the feature matrix and labels into a training set and a testing set, with 20% of the data used for testing.

`LogisticRegression` is a classification algorithm that learns to classify data by finding the coefficients that best separate the classes. The `max_iter` parameter specifies the maximum number of iterations for the solver to converge.
`clf.fit(X_train, y_train)` trains the logistic regression model using the training data.

`clf.score(X_test, y_test)` evaluates the accuracy of the model on the test set by predicting the labels for the test data and comparing them to the true labels. The accuracy is printed to the console.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

verbs = load_corpus()[0]
vectorizer = CountVectorizer(lowercase=False)
X = vectorizer.fit_transform(verbs)

y = ['infinitive' if verb.endswith('el') else 'non-infinitive' for verb in verbs]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)

print('test accuracy =', clf.score(X_test, y_test))

import pandas as pd

feature_names = list(vectorizer.vocabulary_.keys())

coef = clf.coef_[0]

feature_importance = pd.DataFrame(list(zip(feature_names, coef)), columns=['feature', 'importance'])
feature_importance = feature_importance.sort_values(by=['importance'], ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importance[:20], color='b')
plt.title('Top 20 Features by Importance Score')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

import os
import pyconll
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import transliterate

# Load the Armenian TDP data and extract relevant features
corpus_dir = '/content/drive/MyDrive/UD_Armenian-ArmTDP-master'
verbs = []
features = []
for file in os.listdir(corpus_dir):
    if file.endswith('.conllu'):
        data = pyconll.load_from_file(os.path.join(corpus_dir, file))
        for sentence in data:
          for token in sentence:
            if token.upos == 'VERB':
              verb_latin = transliterate.translit(token.lemma, 'hy', reversed=True)
              verbs.append(verb_latin)
              features.append(token.feats)

# Label the data based on the morphological processes that are used in each verb
labels = []
for feat in features:
    if feat is not None:
        if 'Subcat' in feat:
            labels.append('Subcat')
        elif 'Tense' in feat:
            labels.append('Tense')
        elif 'Mood' in feat:
            labels.append('Mood')
        elif 'Voice' in feat:
            labels.append('Voice')
        else:
            labels.append('Other')
    else:
        labels.append('Other')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(verbs, labels, test_size=0.2, random_state=42)

# Vectorize the data using CountVectorizer or Tf-idfVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a machine learning model on the training data
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_vec, y_train)

# Evaluate the performance of the model on the testing data
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Interpret the results and extract the most important features
feature_names = list(vectorizer.vocabulary_.keys())
coef = clf.coef_

for i, label in enumerate(clf.classes_):
    top_features = sorted(zip(feature_names, coef[i]), key=lambda x: -abs(x[1]))[:10]
    print(f'Top features for label "{label}":')
    for feature in top_features:
        print(feature)
