print("RUNNING...")
#import tensorflow as tf
print("IMPORTING NUMPY...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer as TV
import time


print("IMPORTING SEQUENTIAL...")
from keras.models import Sequential
print("IMPORTING LAYERS...")
from keras.layers import Dense, BatchNormalization, Dropout
print("IMPORTING OPTIMIZERS...")
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
print("IMPORTING CALLBACKS...")
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier
from math import floor
from sklearn.metrics import make_scorer, fbeta_score  # Import fbeta_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU
from keras.utils import custom_object_scope

LeakyReLU = LeakyReLU(alpha=0.1)
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

score_f2 = make_scorer(fbeta_score, beta=2)  # Use F2 score

data_train = pd.read_csv('training_preprocessed.csv')
data_test = pd.read_csv('testing_preprocessed.csv')

# Separates training and testing values
X_train = data_train["post_text"]
y_train = data_train["label"].values
X_test = data_test["post_text"]
y_test = data_test["label"].values

# Gets rid of empty rows in the training and testing datasets
data_train = data_train[data_train['post_text'].notnull()]
data_test = data_test[data_test['post_text'].notnull()]

print("VECTORIZING...")
tfidf = TV(max_features= 2000, strip_accents = "ascii", min_df = 2)
X_train = tfidf.fit_transform(X_train.values.astype('U')).toarray()
X_test = tfidf.transform(X_test.values.astype('U')).toarray()

start = time.time()

def f2_score(y_true, y_pred):
    # Convert predictions to binary
    y_pred = np.round(y_pred)

    # Calculate true positives, false positives, and false negatives
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))

    # Avoid division by zero
    epsilon = 1e-10  # Small constant to avoid division by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F2 score
    f2 = 5 * (precision * recall) / (4 * precision + recall + epsilon)
    return np.mean(f2)


print("RUNNING BAYESIAN OPTIMIZATION...")
def nn_cl_bo2(neurons, activation, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, normalization, dropout, dropout_rate):
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','SGD']
    optimizerD= {'Adam': Adam, 'SGD': SGD, 'RMSprop': RMSprop, 'Adadelta': Adadelta,
                 'Adagrad': Adagrad, 'Adamax': Adamax, 'Nadam': Nadam, 'Ftrl': Ftrl}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU, 'relu']
    
    # Round values appropriately
    neurons = round(neurons)
    activation = activationL[round(activation)]
    optimizer_name = optimizerL[round(optimizer)]
    batch_size = round(batch_size)
    epochs = round(epochs)
    layers1 = round(layers1)
    layers2 = round(layers2)

    # Define the neural network building function
    def nn_cl_fun():
        nn = Sequential()
        nn.add(Dense(neurons, input_dim=2000, activation=activation))
        if normalization > 0.5:
            nn.add(BatchNormalization())
        for i in range(layers1):
            nn.add(Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate, seed=123))
        for i in range(layers2):
            nn.add(Dense(neurons, activation=activation))
        nn.add(Dense(1, activation='sigmoid'))
        # Create a fresh optimizer instance inside the model-building function
        opt = optimizerD[optimizer_name](learning_rate=learning_rate)
        # Compile the model
        nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return nn
    # Cross-validation with a fresh model and optimizer for each fold
    def cross_val_with_new_model():
        # Initialize the early stopping callback
        es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
        # KerasClassifier with a fresh model in each fold
        nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
        # Perform Stratified K-Fold Cross-validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        score = cross_val_score(nn, X_train, y_train, scoring=score_f2, cv=kfold, fit_params={'callbacks': [es]}).mean()
        return score

    return cross_val_with_new_model()


params_nn2 ={
    'neurons': (10, 100),
    'activation':(0, 9),
    'optimizer':(0,7),
    'learning_rate':(0.01, 1),
    'batch_size':(200, 1000),
    'epochs':(20, 100),
    'layers1':(1,3),
    'layers2':(1,3),
    'normalization':(0,1),
    'dropout':(0,1),
    'dropout_rate':(0,0.3)
}
# Run Bayesian Optimization
nn_bo = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=111)
nn_bo.maximize(init_points=25, n_iter=50)

params_nn_ = nn_bo.max['params']
learning_rate = params_nn_['learning_rate']
activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
               'elu', 'exponential', LeakyReLU,'relu']
params_nn_['activation'] = activationL[round(params_nn_['activation'])]
params_nn_['batch_size'] = round(params_nn_['batch_size'])
params_nn_['epochs'] = round(params_nn_['epochs'])
params_nn_['layers1'] = round(params_nn_['layers1'])
params_nn_['layers2'] = round(params_nn_['layers2'])
params_nn_['neurons'] = round(params_nn_['neurons'])
optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','Adam']
optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
             'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
             'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
             'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
params_nn_['optimizer'] = optimizerD[optimizerL[round(params_nn_['optimizer'])]]
print(params_nn_)

print('It took %s minutes to perform the Bayesian optimization.' % ((time.time() - start)/60))

start = time.time()

print("FITTING NEURAL NETWORK...")
def nn_cl_fun():
    nn = Sequential()
    nn.add(Dense(params_nn_['neurons'], input_dim=2000, activation=params_nn_['activation']))
    if params_nn_['normalization'] > 0.5:
        nn.add(BatchNormalization())
    for i in range(params_nn_['layers1']):
        nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
    if params_nn_['dropout'] > 0.5:
        nn.add(Dropout(params_nn_['dropout_rate'], seed=123))
    for i in range(params_nn_['layers2']):
        nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer=params_nn_['optimizer'], metrics=['accuracy'])
    return nn


es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
nn = KerasClassifier(build_fn=nn_cl_fun, epochs=params_nn_['epochs'], batch_size=params_nn_['batch_size'],
                        verbose=0)
nn.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1)

print('It took %s minutes to fit the model.' % ((time.time() - start)/60))

print("SAVING MODEL...")

model = nn.model
model.save('mlpbestmodel.keras')
print("I'm finally done! I've been at it for about 10 hours but looks like all the hard work paid off. Congratulations on building the best possible model! It has been saved for you to enjoy.")