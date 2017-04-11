from datetime import date
from functools import lru_cache

import pandas as pd
import vk_api
from numpy import NaN


class RawUsersData:
    def __init__(self, vk_session, group_id):
        self.vk_session = vk_session
        self.vk = self.vk_session.get_api()
        self.vk_tools = vk_api.VkTools(self.vk_session)
        self.group_id = group_id
        self.members = []
        self.member_friends = []
        self.member_fields = 'sex,bdate,country'
        self.group_fields = 'description'

    def fetch(self):
        self._fetch_members()
        self._fetch_members_friends()
        self._fetch_groups()

    def _fetch_members(self):
        self.members = self.vk_tools.get_all('groups.getMembers', 1000,
                                             {'group_id': self.group_id, 'fields': self.member_fields})['items']

    def _fetch_members_friends(self):
        member_friends = []
        for member in self.members:
            friends = self.vk.friends.get(user_id=member['id'], fields=self.member_fields)['items']
            member_friends.extend(friends)

        self.member_friends = []
        for member in member_friends:
            found = False
            for user in self.members:
                if user['id'] == member['id']:
                    found = True
                    break
            if not found:
                self.member_friends.append(member)

    def _fetch_groups(self):
        for users in [self.members, self.member_friends]:
            for user in users:
                user['groups'] = self.vk.groups.get(user_id=user['id'], count=1000, extended=1, fields=self.group_fields)['items']

    def _find_user(self, user_id):
        for user in self.members:
            if user['id'] == user_id:
                return {'user': user, 'is_member': True}
        for user in self.member_friends:
            if user['id'] == user_id:
                return {'user': user, 'is_member': False}
        return None


class TableUsersData:
    def __init__(self, raw_users_data, lda_maker):
        self.raw_users_data = raw_users_data
        self.lda_maker = lda_maker

    def get_all(self, do_members=True, do_member_friends=True):
        users = []
        if do_members:
            users.extend(self.raw_users_data.members)
        if do_member_friends:
            users.extend(self.raw_users_data.member_friends)
        return pd.DataFrame([self.get_row(user) for user in users], index=[user['id'] for user in users])

    @lru_cache(maxsize=-1)
    def get_row(self, user):
        return ([self._user_is_woman(user), self._user_is_man(user), self._user_age(user),
                self._user_is_in_russia(user), self._user_is_in_ukraine(user), self._user_is_in_byelorussia(user), self._user_is_in_kazakstan(user)] +
                self._user_lda_by_groups(user))

    @staticmethod
    def _user_is_woman(user):
        return user['sex'] == 1

    @staticmethod
    def _user_is_man(user):
        return user['sex'] == 2

    @staticmethod
    def _user_age(user):
        if 'bdate' in user:
            bdate_parts = user['bdate'].split('.')
            if len(bdate_parts) == 3:
                year = int(bdate_parts[-1])
                return date.today().year - year
            else:
                return NaN
        else:
            return NaN

    @staticmethod
    def _user_is_in_russia(user):
        return user['country']['id'] == 1

    @staticmethod
    def _user_is_in_ukraine(user):
        return user['country']['id'] == 2

    @staticmethod
    def _user_is_in_byelorussia(user):
        return user['country']['id'] == 3

    @staticmethod
    def _user_is_in_kazakstan(user):
        return user['country']['id'] == 4

    def _user_lda_by_groups(self, user):
        row = [0] * self.lda_maker.num_topics
        for group in user['groups']:
            lda_desc = self.lda_maker.get(group['description'])
            for k, v in lda_desc:
                row[k] = v
        return row
