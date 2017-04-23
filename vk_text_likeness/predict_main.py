import pickle
import random
from collections import Counter

import pandas as pd
import vk_api
from tqdm import tqdm

from vk_text_likeness.action_data import ActionData
from vk_text_likeness.logs import log_method_begin, log_method_end
from vk_text_likeness.predict_model import PredictActionModel, PredictStatsModel
from vk_text_likeness.users_data import RawUsersData, TableUsersData
from vk_text_likeness.wall_data import RawWallData, TableWallData


class GroupPredict:
    def __init__(self, group_id, vk_access_token):
        print('GroupPredict.__init__ for group {}'.format(group_id))

        self.group_id = group_id
        self.vk_session = vk_api.VkApi(token=vk_access_token)

    def prepare(self):
        print('GroupPredict.prepare for group {}'.format(self.group_id))
        self._init_raw_users_data()
        self._init_raw_wall_data()
        self._init_raw_users_data_more()
        self._init_table_users_data()
        self._init_table_wall_data()
        self._init_action_data()

    def fit(self, post_subset=None):
        print('GroupPredict.fit for group {}'.format(self.group_id))
        self._init_predict_action_model(post_subset)
        self._init_predict_stats_model()

    def _init_raw_users_data(self):
        self.raw_users_data = RawUsersData(self.group_id, self.vk_session)

        self.raw_users_data.members = self._try_load_pickle('raw_users_data.members')

        if self.raw_users_data.members is None:
            self.raw_users_data.fetch()

            self._save_pickle('raw_users_data.members', self.raw_users_data.members)

    def _init_raw_wall_data(self):
        self.raw_wall_data = RawWallData(self.group_id, self.vk_session)

        self.raw_wall_data.posts = self._try_load_pickle('raw_wall_data.posts')

        if self.raw_wall_data.posts is None:
            self.raw_wall_data.fetch()

            self._save_pickle('raw_wall_data.posts', self.raw_wall_data.posts)

    def _init_raw_users_data_more(self):
        self.raw_users_data.member_friends = self._try_load_pickle('raw_users_data.member_friends')

        if self.raw_users_data.member_friends is None or self.raw_users_data.was_fetch_groups_error():
            random.seed(42)
            self.raw_users_data.fetch_more(self.raw_wall_data.get_who_liked(), self.raw_wall_data.get_who_reposted())

            self._save_pickle('raw_users_data.members', self.raw_users_data.members)
            self._save_pickle('raw_users_data.member_friends', self.raw_users_data.member_friends)

            if self.raw_users_data.was_fetch_groups_error():
                exit(1)

    def _init_table_users_data(self):
        self.table_users_data = TableUsersData(self.raw_users_data)

        self.table_users_data.lda_maker = self._try_load_pickle('table_users_data.lda_maker')

        if self.table_users_data.lda_maker is None:
            self.table_users_data.fit()

            self._save_pickle('table_users_data.lda_maker', self.table_users_data.lda_maker)

    def _init_table_wall_data(self):
        self.table_wall_data = TableWallData(self.raw_wall_data)

        self.table_wall_data.lda_maker = self._try_load_pickle('table_wall_data.lda_maker')

        if self.table_wall_data.lda_maker is None:
            self.table_wall_data.fit()

            self._save_pickle('table_wall_data.lda_maker', self.table_wall_data.lda_maker)

    def _init_action_data(self):
        self.action_data = ActionData(self.raw_users_data, self.table_users_data, self.raw_wall_data, self.table_wall_data)

        self.action_data.table = self._try_load_pickle('action_data.table')

        if self.action_data.table is None:
            self.action_data.fit()

            self._save_pickle('action_data.table', self.action_data.table)

    def _init_predict_action_model(self, post_subset):
        self.predict_action_model = PredictActionModel(self.action_data)

        if post_subset is None:
            self.predict_action_model.like_model = self._try_load_pickle('predict_action_model.like_model')
            self.predict_action_model.repost_model = self._try_load_pickle('predict_action_model.repost_model')
            self.predict_action_model.is_fitted = True

        if not self.predict_action_model.is_fitted:
            self.predict_action_model = PredictActionModel(self.action_data)
            self.predict_action_model.fit(post_subset)

            if post_subset is None:
                self._save_pickle('predict_action_model.like_model', self.predict_action_model.like_model)
                self._save_pickle('predict_action_model.repost_model', self.predict_action_model.repost_model)

    def _init_predict_stats_model(self):
        self.predict_stats_model = PredictStatsModel(self.predict_action_model, self.raw_users_data, self.action_data)

    def _try_load_pickle(self, name):
        try:
            with open('{}{}.pkl'.format(name, self.group_id), 'rb') as f:
                return pickle.load(f)
        except IOError:
            pass
        except Exception as e:
            print('Can\'t load pickle {}:'.format(name), e)
            return None

    def _save_pickle(self, name, obj):
        try:
            with open('{}{}.pkl'.format(name, self.group_id), 'wb') as f:
                pickle.dump(obj, f)
        except IOError as e:
            print('Can\'t save pickle {}:'.format(name), e)

    def predict(self, indexes=None):
        print('GroupPredict.predict for group {}'.format(self.group_id))
        return self.predict_stats_model.predict(indexes)

    def get_true(self, subset=None):
        print('GroupPredict.get_true for group {}'.format(self.group_id))
        log_method_begin()

        direct_likes_count = Counter()
        direct_reposts_count = Counter()
        non_direct_likes_count = Counter()
        non_direct_reposts_count = Counter()

        for post in tqdm(self.raw_wall_data.posts):
            post_id = post['id']
            if subset is not None and post_id not in subset:
                continue

            for user_id in post['likes']['user_ids']:
                user = self.raw_users_data.find_user(user_id)
                if user is None:
                    continue
                if user['is_member']:
                    direct_likes_count[post_id] += 1
                else:
                    non_direct_likes_count[post_id] += 1

            for user_id in post['reposts']['user_ids']:
                user = self.raw_users_data.find_user(user_id)
                if user is None:
                    continue
                if user['is_member']:
                    direct_reposts_count[post_id] += 1
                else:
                    non_direct_reposts_count[post_id] += 1

        post_ids = list(direct_likes_count.keys() | direct_reposts_count.keys() | non_direct_likes_count.keys() | non_direct_reposts_count.keys())
        rows = []
        for post_id in post_ids:
            rows.append([direct_likes_count[post_id], direct_reposts_count[post_id], non_direct_likes_count[post_id], non_direct_reposts_count[post_id]])
        result = pd.DataFrame(rows, index=post_ids, columns=['direct_likes_count', 'direct_reposts_count', 'non_direct_likes_count', 'non_direct_reposts_count'])
        log_method_end()
        return result
