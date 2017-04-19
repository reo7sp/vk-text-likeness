import pandas as pd

from vk_text_likeness.check import check

if __name__ == '__main__':
    predictions_df = pd.read_csv('predictions.csv', index_col=0)
    true_df = pd.read_csv('true.csv', index_col=0)
    check(predictions_df, true_df)
