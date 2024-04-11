from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import Word
nltk.download('wordnet')

from termcolor import colored
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn import set_config
set_config(print_changed_only=False)

print(colored("\nLIBRARIES WERE SUCCESSFULLY IMPORTED...", color="green", attrs=["dark", "bold"]))

# Load datasets
train_set = pd.read_csv("train.csv",
                        encoding="utf-8",
                        engine="python",
                        header=0)

test_set = pd.read_csv("test.csv",
                       encoding="utf-8",
                       engine="python",
                       header=0)

print(colored("\nDATASETS WERE SUCCESSFULLY LOADED...", color="green", attrs=["dark", "bold"]))

# Clean and process dataset
train_set["tweet"] = train_set["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
test_set["tweet"] = test_set["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))

train_set["tweet"] = train_set["tweet"].str.replace('[^\w\s]', '')
test_set["tweet"] = test_set["tweet"].str.replace('[^\w\s]', '')

train_set['tweet'] = train_set['tweet'].str.replace('\d', '')
test_set['tweet'] = test_set['tweet'].str.replace('\d', '')

sw = stopwords.words("english")
train_set['tweet'] = train_set['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
test_set['tweet'] = test_set['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

train_set['tweet'] = train_set['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
test_set['tweet'] = test_set['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

train_set = train_set.drop("id", axis=1)
test_set = test_set.drop("id", axis=1)

# Divide datasets
x = train_set["tweet"]
y = train_set["label"]

train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=11)

print(colored("\nDIVIDED SUCCESSFULLY...", color="green", attrs=["dark", "bold"]))

# Vectorize data
vectorizer = CountVectorizer()
vectorizer.fit(train_x)

x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)

tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# Build machine learning models
log = linear_model.LogisticRegression()
log_model = log.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(log_model,
                                           x_test_count,
                                           test_y,
                                           cv=20).mean()

print(colored("\nLogistic regression model with 'count-vectors' method", color="red", attrs=["dark", "bold"]))
print(colored("Accuracy ratio: ", color="red", attrs=["dark", "bold"]), accuracy)

log = linear_model.LogisticRegression()
log_model = log.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(log_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv=20).mean()

print(colored("\nLogistic regression model with 'tf-idf' method", color="red", attrs=["dark", "bold"]))
print(colored("Accuracy ratio: ", color="red", attrs=["dark", "bold"]), accuracy)

xgb = XGBClassifier()
xgb_model = xgb.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_count,
                                           test_y,
                                           cv=20).mean()

print(colored("\nXGBoost model with 'count-vectors' method", color="red", attrs=["dark", "bold"]))
print(colored("Accuracy ratio: ", color="red", attrs=["dark", "bold"]), accuracy)

xgb = XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv=20).mean()

print(colored("\nXGBoost model with 'tf-idf' method", color="red", attrs=["dark", "bold"]))
print(colored("Accuracy ratio: ", color="red", attrs=["dark", "bold"]), accuracy)

lgbm = LGBMClassifier()
lgbm_model = lgbm.fit(x_train_count.astype("float64"), train_y)
accuracy = model_selection.cross_val_score(lgbm_model,
                                           x_test_count.astype("float64"),
                                           test_y,
                                           cv=20).mean()

print(colored("\nLight GBM model with 'count-vectors' method", color="red", attrs=["dark", "bold"]))
print(colored("Accuracy ratio: ", color="red", attrs=["dark", "bold"]), accuracy)

lgbm = LGBMClassifier()
lgbm_model = lgbm.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(lgbm_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv=20).mean()

print(colored("\nLight GBM model with 'tf-idf' method", color="red", attrs=["dark", "bold"]))
print(colored("Accuracy ratio: ", color="red", attrs=["dark", "bold"]), accuracy)

# ROC AUC (curvature)
y = train_y
X = x_train_count.astype("float64")

logit_roc_auc = roc_auc_score(y, lgbm_model.predict(X))

fpr, tpr, thresholds = roc_curve(y, lgbm_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()

# Estimation over test
# Estimation over test set
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
test_set_encoded = vectorizer.transform(test_set["tweet"])

predicted_labels = lgbm_model.predict(test_set_encoded.astype("float"))

print(predicted_labels[:5])  # Sample of predicted labels

# Visualization with Word Cloud
from wordcloud import WordCloud

tw_mask = np.array(Image.open('twitter_mask3.jpg'))

text = " ".join(i for i in train_set.tweet)

wc = WordCloud(background_color="white",
               width=600, mask=tw_mask,
               height=600,
               contour_width=0,
               contour_color="red",
               max_words=1000,
               scale=1,
               collocations=False,
               repeat=True,
               min_font_size=1)

wc.generate(text)

plt.figure(figsize=[15, 15])
plt.imshow(wc)
plt.axis("off")
plt.show()