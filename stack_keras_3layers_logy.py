''' 
Author: Danijel Kivaranovic 
Title: Neural network (Keras) with sparse data
'''

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
path = '/home/jun/Documents/Kaggle/allstate_claim_severity/'

train = pd.read_csv(path + 'data/train.csv')
test = pd.read_csv(path + 'data/test.csv')
split = pd.read_csv(path + 'data/split_ind.csv')

train.shape, test.shape, split.shape

id_test = test['id'].values

train_tv = pd.merge(train, split, how='inner', on=['id'])

train_tv.split.value_counts()

train_tv.sort_values(by = 'split', inplace=True)

id_train = train_tv['id'][train_tv['split']==0]
id_hold = train_tv['id'][train_tv['split']==1]

y_train = train_tv['loss'][train_tv['split']==0]
y_hold = train_tv['loss'][train_tv['split']==1]

y_train = np.log(y_train.values + 200)
y_hold = np.log(y_hold.values + 200)

ntrain = train.shape[0]
nt = id_train.shape[0]

train = train_tv.drop('split', axis=1)
test['loss'] = np.nan
train.shape, test.shape

## stack train test

tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtv = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

xtrain = xtv[:nt, :]
xhold = xtv[nt:, :]

print(xtrain.shape, xhold.shape, xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

## cv-folds
nfolds = 10
folds = KFold(len(y_train), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 10
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_hold = np.zeros(xhold.shape[0])
pred_test = np.zeros(xtest.shape[0])

earlyStopping=EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y_train[inTr]
    xte = xtrain[inTe]
    yte = y_train[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  callbacks=[earlyStopping],
                                  samples_per_epoch = xtr.shape[0],
                                  validation_data=(xhold.todense(), y_hold),
                                  verbose = 1)
        pred += np.exp(model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0])-200
        pred_hold += np.exp(model.predict_generator(generator = batch_generatorp(xhold, 800, False), val_samples = xhold.shape[0])[:,0])-200
        pred_test += np.exp(model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0])-200
    pred /= nbags   
    pred_oob[inTe] = pred
    fold_mae = mean_absolute_error(np.exp(yte)-200, pred)
    hold_mae = mean_absolute_error(np.exp(y_hold)-200, pred_hold/nbags)  
    i += 1
    print('Fold ', i, '- MAE:', fold_mae)
    print('Hold - MAE:', hold_mae)
   
print('Total - Fold MAE:', mean_absolute_error(np.exp(y_train)-200, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv(path + 'cache/stack_keras_3layers_logy_oob.csv', index = False)

## hold predictions
pred_hold /= (nfolds*nbags)

hold_mae = mean_absolute_error(np.exp(y_hold)-200, pred_hold)
print('Total - Hold MAE:', hold_mae)

df = pd.DataFrame({'id': id_hold, 'loss': pred_hold})
df.to_csv(path + 'cache/stack_keras_3layers_logy_hold.csv', index = False)


## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv(path + 'cache/stack_keras_3layers_logy_test.csv', index = False)












