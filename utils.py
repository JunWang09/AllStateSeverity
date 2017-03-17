import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import sparse
import cPickle as pickle
import operator
import matplotlib.pyplot as plt

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
              
def JExploreDataAnalysis(table, outfilename):
    totRow = table.shape[0]
    # logger.info("check all variable distributions")
    cat = table.dtypes == 'object'
    cat = cat[cat].index.tolist()
    cat_dist = pd.DataFrame()
    for var in cat:
        freq = table[var].value_counts().head(10)
        freqIndex = list(freq.index)
        count = list(freq)
        percent = list((100*freq/totRow).map('{0:.1f}%'.format))
        
        if len(freq) < 10:
            freqIndex = list(freq.index) + ['.']*(10-len(freq))
            count = list(freq) + ['.']*(10-len(freq))
            percent = list((100*freq/totRow).map('{0:.1f}%'.format)) + ['.']*(10-len(freq)) 
            
        cat_dist[var + ' Top10'] = freqIndex
        cat_dist['# ' + var] = count
        cat_dist['% ' + var] = percent
        
    # logger.info("check continuous variable distribution")
    num = table.dtypes != 'object'
    num = num[num].index.tolist()
    
    num_dist = pd.DataFrame(['count', 'mean', 'std', 'min', '1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%', 'max'], columns = (['Percentile']))
    for var in num:
        quant = table[var].describe(percentiles=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99])
        num_dist[var] = list(quant)
        
    writer = pd.ExcelWriter(path + outfilename +'.xlsx')
    cat_dist.to_excel(writer, 'Categorical Distributions')
    num_dist.to_excel(writer, 'Numerical Distributions')
    writer.save()
    
def save_data(filename, y_train,X_train, X_test, id_test, hold):
    #logger.info(">>>saving %s to disk..." % filename)
    with open(path + "cache/%s.pkl" % filename, 'wb') as f:
        pickle.dump((y_train, X_train, X_test, id_test, hold), f, pickle.HIGHEST_PROTOCOL)
    
def get_data(filename):
    try:
        with open(path + "cache/%s.pkl" % filename, 'rb') as f:
            y_train, X_train, X_test, id_test, hold = pickle.load(f)
    except IOError:
        #logging.warning("could not find table %s", filename)
        return False
    return y_train, X_train, X_test, id_test, hold


def feature_map(features, filename):
    outfile = open(path + 'fmap/%s.fmap' % filename, 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def ImpVars(fmap, xgb_model, lvl=50):
    fscore = xgb_model.get_fscore(path + 'fmap/%s.fmap' % fmap)
    fscore = sorted(fscore.items(), key=operator.itemgetter(1,0), reverse=True)
    df = pd.DataFrame(fscore, columns=['feature', 'fscore'])
    #df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 20))
    #plt.title('XGBoost Feature Importance')
    #plt.xlabel('relative importance')
    #plt.savefig(path + 'plot/VarImportance_%s.png' % fmap, bbox_inches='tight', pad_inches=1)
    #imp_vars = df[df['fscore']>lvl]
    return df    

    
    
     
