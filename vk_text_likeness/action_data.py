import pandas as pd
from tqdm import tqdm

from vk_text_likeness.logs import log_method_begin, log_method_end


class ActionData:
    def __init__(self, raw_users_data, table_users_data, raw_wall_data, table_wall_data):
        self.raw_users_data = raw_users_data
        self.table_users_data = table_users_data
        self.raw_wall_data = raw_wall_data
        self.table_wall_data = table_wall_data

    def get_all(self):
        log_method_begin()
        print("{} members, {} posts".format(len(self.raw_users_data.members), len(self.raw_wall_data.posts)))

        rows = []

        friend_post_pairs = set()
        for user in tqdm(self.raw_users_data.members, 'ActionData.get_all: for members'):
            if 'groups' not in user:
                continue
            for post in self.raw_wall_data.posts:
                is_liked = user['id'] in post['likes']['user_ids']
                is_reposted = user['id'] in post['reposts']['user_ids']

                if is_reposted:
                    for friend in self.raw_users_data.member_friends[user['id']]:
                        if 'groups' not in friend:
                            continue
                        friend_post_pair = (friend['id'], post['id'])
                        if friend_post_pair not in friend_post_pairs:
                            friend_is_liked = friend['id'] in post['likes']['user_ids']
                            friend_is_reposted = friend['id'] in post['reposts']['user_ids']

                            rows.append(self.get_row(friend, post, False, friend_is_liked, friend_is_reposted))
                            friend_post_pairs.add(friend_post_pair)

                rows.append(self.get_row(user, post, True, is_liked, is_reposted))

        result = pd.DataFrame(rows, columns=self.get_labels())
        print("{} rows".format(len(result)))
        print("{} liked, {} reposted".format(
            sum(result['is_liked']), sum(result['is_reposted'])
        ))
        print("{} liked, {} reposted by members".format(
            sum(result[result['is_member']]['is_liked']), sum(result[result['is_member']]['is_reposted'])
        ))
        log_method_end()
        return result

    def get_row(self, user, post, is_member, is_liked, is_reposted):
        return [user['id']] + self.table_users_data.get_row(user) + \
               [post['id']] + self.table_wall_data.get_row(post) + \
               [is_member, is_liked, is_reposted]

    def get_labels(self):
        return ['user_id'] + self.table_users_data.get_labels() + \
               ['post_id'] + self.table_wall_data.get_labels() + \
               ['is_member', 'is_liked', 'is_reposted']
