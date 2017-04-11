import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class PredictActionModel:
    def __init__(self, predict_data):
        self.predict_data = predict_data
        self.like_model = RandomForestClassifier()
        self.repost_model = RandomForestClassifier()

    def fit(self):
        df = self.predict_data.get_all()
        x_df = df.drop(['user_id', 'post_id', 'is_liked', 'is_reposted'])
        self.like_model.fit(x_df, df['is_liked'])
        self.repost_model.fit(x_df, df['is_reposted'])

    def predict(self, df):
        x_df = df.drop(['user_id', 'post_id', 'is_liked', 'is_reposted'])
        pred = np.hstack([df['user_id'], df['post_id'], self.like_model.predict(x_df), self.repost_model.predict(x_df)])
        return pd.DataFrame(pred, columns=['user_id', 'post_id', 'is_liked', 'is_reposted'])


class PredictStatsModel:
    def __init__(self, predict_action_model, raw_users_data):
        self.predict_action_model = predict_action_model
        self.raw_users_data = raw_users_data

    def predict(self, df):
        direct_likes_count = 0
        reposts_count = 0
        non_direct_likes_count = 0

        pred_df = self.predict_action_model.predict(df)
        member_ids = set(user['id'] for user in self.raw_users_data.members)
        member_friend_ids = set(user['id'] for user in self.raw_users_data.member_friends)
        for row in pred_df.iterrows():
            if row['is_liked']:
                if row['user_id'] in member_ids:
                    direct_likes_count += 1
                if row['user_id'] in member_friend_ids:
                    non_direct_likes_count += 1
            if row['is_reposted']:
                if row['user_id'] in member_ids:
                    reposts_count += 1

        return {'direct_likes_count': direct_likes_count,
                'reposts_count': reposts_count,
                'non_direct_likes_count': non_direct_likes_count}
