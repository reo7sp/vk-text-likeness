from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class PredictActionModel:
    def __init__(self, action_data):
        self.action_data = action_data
        self.like_model = RandomForestClassifier()
        self.repost_model = RandomForestClassifier()

    def fit(self):
        df = self.action_data.get_all()
        x_df = df.drop(['user_id', 'post_id', 'is_member', 'is_liked', 'is_reposted'], axis=1)
        self.like_model.fit(x_df, df['is_liked'])
        self.repost_model.fit(x_df, df['is_reposted'])

    def predict(self):
        df = self.action_data.get_all()
        x_df = df.drop(['user_id', 'post_id', 'is_member', 'is_liked', 'is_reposted'], axis=1)
        pred = [df['user_id'], df['post_id'], df['is_member'], self.like_model.predict(x_df), self.repost_model.predict(x_df)]
        return pd.DataFrame(np.array(pred).T, columns=['user_id', 'post_id', 'is_member', 'is_liked', 'is_reposted'])


class PredictStatsModel:
    def __init__(self, predict_action_model, raw_users_data, action_data):
        self.predict_action_model = predict_action_model
        self.raw_users_data = raw_users_data
        self.action_data = action_data

    def predict(self):
        direct_likes_count = Counter()
        direct_reposts_count = Counter()
        non_direct_likes_count = Counter()
        non_direct_reposts_count = Counter()

        pred_df = self.predict_action_model.predict()
        for i, row in pred_df.iterrows():
            if row['is_liked']:
                if row['is_member']:
                    direct_likes_count[row['post_id']] += 1
                else:
                    non_direct_likes_count[row['post_id']] += 1
            if row['is_reposted']:
                if row['is_member']:
                    direct_reposts_count[row['post_id']] += 1
                else:
                    non_direct_reposts_count[row['post_id']] += 1

        post_ids = list(direct_likes_count.keys() | direct_reposts_count.keys() | non_direct_likes_count.keys() | non_direct_reposts_count.keys())
        rows = []
        for post_id in post_ids:
            rows.append([direct_likes_count[post_id], direct_reposts_count[post_id], non_direct_likes_count[post_id], non_direct_reposts_count[post_id]])
        return pd.DataFrame(rows, index=post_ids, columns=['direct_likes_count', 'direct_reposts_count', 'non_direct_likes_count', 'non_direct_reposts_count'])
