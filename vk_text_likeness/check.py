import numpy as np
from scipy.stats.stats import pearsonr


def check(predictions_df, true_df):
    index = [ind for ind in predictions_df.index if ind in true_df.index]
    for column in ['direct_likes_count', 'direct_reposts_count', 'non_direct_likes_count', 'non_direct_reposts_count']:
        x = predictions_df[column].loc[index]
        y_raw = true_df[column]
        y = [y_raw.loc[ind] for ind in index]
        x = x.values
        y = np.array(y)

        rmse = np.sqrt(np.mean((x - y) ** 2))
        corr, pval = pearsonr(x, y)

        print(column)
        print('\t', 'rmse:', rmse)
        print('\t', 'corr:', corr)
