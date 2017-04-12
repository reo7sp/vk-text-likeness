from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class PredictActionModel:
    def __init__(self, predict_action_data):
        self.predict_action_data = predict_action_data
        self.like_model = RandomForestClassifier()
        self.repost_model = RandomForestClassifier()

    def fit(self):
        df = self.predict_action_data.get_all()
        x_df = df.drop(['user_id', 'post_id', 'is_liked', 'is_reposted'], axis=1)
        self.like_model.fit(x_df, df['is_liked'])
        self.repost_model.fit(x_df, df['is_reposted'])

    def predict(self):
        df = self.predict_action_data.get_all()
        x_df = df.drop(['user_id', 'post_id', 'is_liked', 'is_reposted'], axis=1)
        pred = [df['user_id'], df['post_id'], self.like_model.predict(x_df), self.repost_model.predict(x_df)]
        return pd.DataFrame(np.array(pred).T, columns=['user_id', 'post_id', 'is_liked', 'is_reposted'])


class PredictStatsModel:
    def __init__(self, predict_action_model, raw_users_data, predict_action_data):
        self.predict_action_model = predict_action_model
        self.raw_users_data = raw_users_data
        self.predict_action_data = predict_action_data

    def predict(self):
        direct_likes_count = Counter()
        reposts_count = Counter()
        non_direct_likes_count = Counter()

        pred_df = self.predict_action_model.predict()
        member_ids = set(user['id'] for user in self.raw_users_data.members)
        member_friend_ids = set(user['id'] for user in self.raw_users_data.member_friends)
        for i, row in pred_df.iterrows():
            if row['is_liked']:
                if row['user_id'] in member_ids:
                    direct_likes_count[row['post_id']] += 1
                if row['user_id'] in member_friend_ids:
                    non_direct_likes_count[row['post_id']] += 1
            if row['is_reposted']:
                if row['user_id'] in member_ids:
                    reposts_count[row['post_id']] += 1

        post_ids = list(direct_likes_count.keys() | reposts_count.keys() | non_direct_likes_count.keys())
        rows = []
        for post_id in post_ids:
            rows.append([direct_likes_count[post_id], reposts_count[post_id], non_direct_likes_count[post_id]])
        return pd.DataFrame(rows, index=post_ids, columns=['direct_likes_count', 'reposts_count', 'non_direct_likes_count'])
