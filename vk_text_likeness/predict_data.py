from collections import defaultdict

import pandas as pd
from tqdm import tqdm


class PredictActionData:
    def __init__(self, raw_users_data, table_users_data, raw_wall_data, table_wall_data):
        self.raw_users_data = raw_users_data
        self.table_users_data = table_users_data
        self.raw_wall_data = raw_wall_data
        self.table_wall_data = table_wall_data

    def get_all(self):
        rows = []
        for post in tqdm(self.raw_wall_data.posts):
            user_actions = defaultdict(lambda: {'is_liked': False, 'is_reposted': False})
            for user in post['likes']['users']:
                user_actions[user['id']]['is_liked'] = True
            for user in post['reposts']['users']:
                user_actions[user['id']]['is_reposted'] = True
            for user_id, actions in user_actions.items():
                user = self.raw_users_data.find_user(user_id)['user']
                rows.append(self.get_row(user, post, actions['is_liked'], actions['is_reposted']))
        return pd.DataFrame(rows, columns=self.get_labels())

    def get_row(self, user, post, is_liked, is_reposted):
        return [user['id']] + self.table_users_data.get_row(user) + \
               [post['id']] + self.table_wall_data.get_row(post) + \
               [is_liked, is_reposted]

    def get_labels(self):
        return ['user_id'] + self.table_users_data.get_labels() + \
               ['post_id'] + self.table_wall_data.get_labels() + \
               ['is_liked', 'is_reposted']

