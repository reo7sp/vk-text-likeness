import pickle

import vk_api

from vk_text_likeness.action_data import ActionData
from vk_text_likeness.predict_model import PredictActionModel, PredictStatsModel
from vk_text_likeness.users_data import RawUsersData, TableUsersData
from vk_text_likeness.wall_data import RawWallData, TableWallData


class GroupPredict:
    def __init__(self, group_id, vk_access_token):
        print('=====> GroupPredict.__init__', group_id)
        self.group_id = group_id

        self.vk_session = vk_api.VkApi(token=vk_access_token)

        self._init_raw_users_data(group_id)
        self._init_table_users_data(group_id)
        self._init_raw_wall_data(group_id)
        self._init_table_wall_data(group_id)
        self._init_action_data()
        self._init_predict_action_model(group_id)
        self._init_predict_stats_model()

    def _init_raw_users_data(self, group_id):
        self.raw_users_data = RawUsersData(group_id, self.vk_session)

        self.raw_users_data.members = self._try_load_pickle('raw_users_data.members')
        self.raw_users_data.member_friends = self._try_load_pickle('raw_users_data.member_friends')

        if self.raw_users_data.members is None or self.raw_users_data.member_friends is None:
            print('-----> raw_users_data.fetch', group_id)
            self.raw_users_data.fetch()

            self._save_pickle('raw_users_data.members', self.raw_users_data.members)
            self._save_pickle('raw_users_data.member_friends', self.raw_users_data.member_friends)

    def _init_table_users_data(self, group_id):
        self.table_users_data = TableUsersData(self.raw_users_data)

        self.table_users_data.lda_maker = self._try_load_pickle('table_users_data.lda_maker')

        if self.table_users_data.lda_maker is None:
            print('-----> table_users_data.fit', group_id)
            self.table_users_data.fit()

            self._save_pickle('table_users_data.lda_maker', self.table_users_data.lda_maker)

    def _init_raw_wall_data(self, group_id):
        self.raw_wall_data = RawWallData(group_id, self.vk_session)

        self.raw_wall_data.posts = self._try_load_pickle('raw_wall_data.posts')

        if self.raw_wall_data.posts is None:
            print('-----> raw_wall_data.fetch', group_id)
            self.raw_wall_data.fetch()

            self._save_pickle('raw_wall_data.posts', self.raw_wall_data.posts)

    def _init_table_wall_data(self, group_id):
        self.table_wall_data = TableWallData(self.raw_wall_data)

        self.table_wall_data.lda_maker = self._try_load_pickle('table_wall_data.lda_maker')

        if self.table_wall_data.lda_maker is None:
            print('-----> table_wall_data.fit', group_id)
            self.table_wall_data.fit()

            self._save_pickle('table_wall_data.lda_maker', self.table_wall_data.lda_maker)

    def _init_action_data(self):
        self.action_data = ActionData(self.raw_users_data, self.table_users_data, self.raw_wall_data, self.table_wall_data)

    def _init_predict_action_model(self, group_id):
        self.predict_action_model = PredictActionModel(self.action_data)

        self.predict_action_model.like_model = self._try_load_pickle('predict_action_model.like_model')
        self.predict_action_model.repost_model = self._try_load_pickle('predict_action_model.repost_model')

        if self.predict_action_model.like_model is None or self.predict_action_model.repost_model:
            self.predict_action_model = PredictActionModel(self.action_data)
            print('-----> predict_action_model.fit', group_id)
            self.predict_action_model.fit()

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

    def predict(self):
        print('=====> predict_stats_model.predict()', self.group_id)
        return self.predict_stats_model.predict()
