import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

if __name__ == '__main__':
    predictions = pd.read_csv('predictions.csv', index_col=0)
    true = pd.read_csv('true.csv', index_col=0)

    for column in ['direct_likes_count', 'direct_reposts_count', 'non_direct_likes_count', 'non_direct_reposts_count']:
        index = [ind for ind in predictions.index if ind in true.index]
        x = predictions[column].loc[index]
        y_raw = true[column]
        y = [y_raw.loc[ind] for ind in index]
        x = x.values
        y = np.array(y)
        print(column, '\t', 'rmse', '\t', np.sqrt(np.mean((x - y) ** 2)))
        corr, pval = pearsonr(x, y)
        print(column, '\t', 'corr', '\t', corr)
