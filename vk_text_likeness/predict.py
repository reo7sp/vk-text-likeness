import vk_api

from vk_text_likeness.predict_data import PredictActionData
from vk_text_likeness.predict_model import PredictActionModel, PredictStatsModel
from vk_text_likeness.users_data import RawUsersData, TableUsersData
from vk_text_likeness.wall_data import RawWallData, TableWallData


class GroupPredict:
    def __init__(self, group_id, vk_access_token):
        print('GroupPredict.__init__', group_id)
        self.group_id = group_id

        self.vk_session = vk_api.VkApi(token=vk_access_token)

        self.raw_users_data = RawUsersData(group_id, self.vk_session)
        print('  raw_users_data.fetch()', group_id)
        self.raw_users_data.fetch()
        self.table_users_data = TableUsersData(self.raw_users_data)

        self.raw_wall_data = RawWallData(group_id, self.vk_session)
        print('  raw_wall_data.fetch()', group_id)
        self.raw_wall_data.fetch()
        self.table_wall_data = TableWallData(self.raw_wall_data)

        self.predict_action_data = PredictActionData(self.raw_users_data, self.table_users_data, self.raw_wall_data, self.table_wall_data)
        self.predict_action_model = PredictActionModel(self.predict_action_data)
        print('  predict_action_model.fit()', group_id)
        self.predict_action_model.fit()

        self.predict_stats_model = PredictStatsModel(self.predict_action_model, self.raw_users_data, self.predict_action_data)

    def predict(self):
        print('predict_stats_model.predict()', self.group_id)
        self.predict_stats_model.predict()
