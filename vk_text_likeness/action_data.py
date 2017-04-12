import pandas as pd
from tqdm import tqdm


class ActionData:
    def __init__(self, raw_users_data, table_users_data, raw_wall_data, table_wall_data):
        self.raw_users_data = raw_users_data
        self.table_users_data = table_users_data
        self.raw_wall_data = raw_wall_data
        self.table_wall_data = table_wall_data

    def get_all(self):
        rows = []
        for user in tqdm(self.raw_users_data.members, 'ActionData.get_all: for members'):
            for post in self.raw_wall_data.posts:
                is_liked = False
                is_reposted = False
                for user_id in post['likes']['users']:
                    if user_id == user['id']:
                        is_liked = True
                        break
                for user_id in post['reposts']['users']:
                    if user_id == user['id']:
                        is_reposted = True
                        break

                if is_reposted:
                    for friend in self.raw_users_data.member_friends[user['id']]:
                        friend_is_liked = False
                        friend_is_reposted = False
                        for user_id in post['likes']['users']:
                            if user_id == friend['id']:
                                friend_is_liked = True
                                break
                        for user_id in post['reposts']['users']:
                            if user_id == friend['id']:
                                friend_is_reposted = True
                                break
                        rows.append(self.get_row(friend, post, False, friend_is_liked, friend_is_reposted))

                rows.append(self.get_row(user, post, True, is_liked, is_reposted))

        return pd.DataFrame(rows, columns=self.get_labels())

    def get_row(self, user, post, is_member, is_liked, is_reposted):
        return [user['id']] + self.table_users_data.get_row(user) + \
               [post['id']] + self.table_wall_data.get_row(post) + \
               [is_member, is_liked, is_reposted]

    def get_labels(self):
        return ['user_id'] + self.table_users_data.get_labels() + \
               ['post_id'] + self.table_wall_data.get_labels() + \
               ['is_member', 'is_liked', 'is_reposted']

