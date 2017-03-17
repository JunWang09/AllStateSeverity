import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import sparse
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def cnt(dsn, varlist):
    df = dsn[varlist]
    df['cnt'] = 1.0
    return df.groupby(varlist).transform(np.sum).cnt
        

def loo(dsn, varlist, y, split, r_k=.3):
    
    #median of target as complement
    y0 = np.median(y[split == 0])
    
    # table for all records
    df1 = dsn[varlist]
    df1.loc[:,'y'] = y
    
    # table for training only
    df2 = df1[split == 0]
    df2.loc[:,'cnt'] = 1.0
    
    # calculate grouped total y and count
    grouped = df2.groupby(varlist).sum().add_prefix('sum_')
    # calculate credibility 
    cred_k = grouped.sum_cnt.mean()
    
    # merge back to whole table
    df1 = pd.merge(df1, grouped, left_on = varlist, right_index = True, how = 'left')
    df1.fillna(0, inplace=True)
    
    # leave one out
    df1['sum_cnt'][split == 0] = df1['sum_cnt'][split == 0] - 1
    df1['sum_y'][split == 0] = df1['sum_y'][split == 0] - df1['y'][split == 0]
    
    # calculated transformed cats based on credibilty
    df1['loo_y'] = (df1['sum_y'] + y0*cred_k)*1.0 / (df1['sum_cnt'] + cred_k)
    df1['loo_y'][df1['loo_y'].isnull()] = y0
    
    # add noise to pretent overfitting
    df1['loo_y'][split == 0] = df1['loo_y'][split == 0]*(1+(np.random.uniform(0,1,sum(split == 0))-0.5)*r_k)
    return df1['loo_y']

# train & test include y variable
def loo_cv(train, test, val_index, hold):
    train['split'] = 0
    train['split'][val_index] = 1
    train['split'][hold==1] = 2
    test['split'] = 3
    
    tt = pd.concat((train, test)).reset_index(drop=True)
    tt.shape
    
    cats = tt.dtypes[tt.dtypes == "object"].index
    for var in cats:
        tt[var] = loo(dsn=tt, varlist=[var], y=tt.y, split=tt.split, r_k=.3)
    
    split = tt.split
    tt.drop(['y', 'split'], axis=1, inplace=True)
    X_train_cv = tt[split == 0]
    X_val_cv = tt[split == 1]
    X_hold = tt[split == 2]
    X_test = tt[split == 3]
    return X_train_cv, X_val_cv, X_hold, X_test



def loo_1(dsn, varlist, y, split):
    
    y = np.log(y+200)    
    #median of target as complement
    y0 = np.median(y[split == 0])
    
    # table for all records
    df1 = dsn[varlist]
    df1['y'] = y
    
    # table for training only
    df2 = df1[split == 0]
    df2['cnt'] = 1.0
    
    # calculate grouped total y and count
    grouped1 = df2.groupby(varlist)[['cnt']].sum().add_prefix('sum_')
    grouped2 = df2.groupby(varlist)[['y']].median().add_prefix('median_')
    
    # merge back to whole table
    df1 = pd.merge(df1, grouped1, left_on = varlist, right_index = True, how = 'left')
    df1 = pd.merge(df1, grouped2, left_on = varlist, right_index = True, how = 'left')
    df1.fillna(0, inplace=True)
    
    # calculated transformed cats based on credibilty
    df1['loo_y'] = df1['median_y']
    #df1['loo_y'][df1.sum_cnt <= 3] = y0
    df1['loo_y'][df1['loo_y'].isnull()] = y0
    
    return df1['loo_y']


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds)-200, np.exp(labels)-200)
    
def encode(charcode):
    r = 0
    ln = len(charcode)
    for i in range(ln):
        r += (ord(charcode[i])-ord('A')+1)*26**(ln-i-1)
    return r
    
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds-labels
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess

def mungeskewed(train, test, numeric_feats):
    test['loss'] = 0
    train_test = train.append(test)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test
    

