import pickle
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adagrad
from scikeras.wrappers import KerasClassifier
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import RocCurveDisplay
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def nn_cl_fun():
    nn = Sequential()
    print("ADDING LAYERS...")
    nn.add(Dense(93, input_dim=2000, activation='softplus'))
    nn.add(BatchNormalization())
    nn.add(Dense(93, activation='softplus'))
    nn.add(Dropout(0.12, seed=123))
    for i in range(2):
        nn.add(Dense(93, activation='softplus'))
    nn.add(Dense(1, activation='sigmoid'))
    print("COMPILING...")
    nn.compile(loss='binary_crossentropy', optimizer=Adagrad(learning_rate=0.418), metrics=['recall'])
    return nn

data_train = pd.read_csv('training_preprocessed.csv')
data_test = pd.read_csv('testing_preprocessed.csv')

# Gets rid of empty rows in the training and testing datasets
data_train = data_train[data_train['post_text'].notnull()]
data_test = data_test[data_test['post_text'].notnull()]

# Separates training and testing values
X_train = data_train["post_text"]
y_train = data_train["label"].values
X_test = data_test["post_text"]
y_test = data_test["label"].values

print("VECTORIZING...")
tfidf = TV(max_features= 2000, strip_accents = "ascii", min_df = 2)
X_train = tfidf.fit_transform(X_train.values.astype('U')).toarray()

print("BUILDING MODEL...")
nn = KerasClassifier(build_fn=nn_cl_fun, epochs=86, batch_size=921,
                        verbose=0)
nn.fit(X_train, y_train)

import pickle
pickle.dump(tfidf, open('tfidf.pkl','wb'))

print("DONE COMPILING. SAVING...")
model = nn.model_
model.save('mlpbestmodel.keras')