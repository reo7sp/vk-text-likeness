from collections import defaultdict
from datetime import date

import pandas as pd
import vk_api
from tqdm import tqdm

from vk_text_likeness.lda_maker import LdaMaker
from vk_text_likeness.tools import cache_by_entity_id


class RawUsersData:
    def __init__(self, group_id, vk_session):
        self.group_id = group_id
        self.vk_session = vk_session
        self.vk = self.vk_session.get_api()
        self.vk_tools = vk_api.VkTools(self.vk_session)
        self.members = []
        self.member_friends = defaultdict(lambda: [])
        self.member_fields = 'sex,bdate,country'
        self.group_fields = 'description'

    def fetch(self):
        self._fetch_members()
        self._fetch_members_friends()
        self._fetch_groups()

    def _fetch_members(self):
        self.members = self.vk_tools.get_all('groups.getMembers', 1000,
                                             {'group_id': self.group_id, 'fields': self.member_fields})['items']
        for member in self.members:
            member['is_member'] = True

    def _fetch_members_friends(self):
        self.member_friends = defaultdict(lambda: [])
        member_ids = set(user['id'] for user in self.members)
        for member in tqdm(self.members, 'RawUsersData._fetch_members_friends: for members'):
            try:
                friends = self.vk.friends.get(user_id=member['id'], fields=self.member_fields)['items']
                for friend in friends:
                    if friend['id'] not in member_ids:
                        friend['is_member'] = False
                        self.member_friends[member['id']].append(friend)
            except vk_api.exceptions.ApiError:
                pass

    def _fetch_groups(self):
        for user in tqdm(self.get_all_users(), 'RawUsersData._fetch_groups: for members + member_friends'):
            try:
                user['groups'] = self.vk.groups.get(user_id=user['id'], count=1000, extended=1, fields=self.group_fields)['items']
            except vk_api.exceptions.ApiError:
                pass

    def find_user(self, user_id):
        for user in self.members:
            if user['id'] == user_id:
                return user
        for users in self.member_friends.values():
            for user in users:
                if user['id'] == user_id:
                    return user
        return None

    def get_all_users(self):
        everything = []
        ids = set()
        for users in [self.members] + list(self.member_friends.values()):
            for user in users:
                if user['id'] not in ids:
                    everything.append(user)
                    ids.add(user['id'])
        return everything


class TableUsersData:
    def __init__(self, raw_users_data, num_topics=15):
        self.raw_users_data = raw_users_data
        self.num_topics = num_topics

    def fit(self):
        self.lda_maker = LdaMaker(self._get_corpora_for_lda(), self.num_topics)

    def get_all(self):
        users = self.raw_users_data.get_all_users()
        return pd.DataFrame([self.get_row(user) for user in tqdm(users, 'TableUsersData.get_all: for users')], index=[user['id'] for user in users])

    @cache_by_entity_id
    def get_row(self, user):
        return [self._user_is_woman(user), self._user_is_man(user), self._user_age(user),
                self._user_is_in_russia(user), self._user_is_in_ukraine(user), self._user_is_in_byelorussia(user), self._user_is_in_kazakstan(user)] + \
                self._user_lda_by_groups(user)

    def get_labels(self):
        return (['is_woman', 'is_man', 'age',
                 'is_in_russia', 'is_in_ukraine', 'is_in_byelorussia', 'is_in_kazakstan'] +
                ['user_lda' + str(i) for i in range(self.lda_maker.num_topics)])

    def _get_corpora_for_lda(self):
        corpora = set()
        for user in self.raw_users_data.get_all_users():
            for group in user['groups']:
                try:
                    doc = group['description']
                    corpora.add(doc)
                except KeyError:
                    pass
        return list(corpora)

    @staticmethod
    def _user_is_woman(user):
        if 'sex' not in user:
            return False
        return user['sex'] == 1

    @staticmethod
    def _user_is_man(user):
        if 'sex' not in user:
            return False
        return user['sex'] == 2

    @staticmethod
    def _user_age(user):
        if 'bdate' in user:
            bdate_parts = user['bdate'].split('.')
            if len(bdate_parts) == 3:
                year = int(bdate_parts[-1])
                return date.today().year - year
            else:
                return -1
        else:
            return -1

    @staticmethod
    def _user_is_in_russia(user):
        if 'country' not in user:
            return False
        return user['country']['id'] == 1

    @staticmethod
    def _user_is_in_ukraine(user):
        if 'country' not in user:
            return False
        return user['country']['id'] == 2

    @staticmethod
    def _user_is_in_byelorussia(user):
        if 'country' not in user:
            return False
        return user['country']['id'] == 3

    @staticmethod
    def _user_is_in_kazakstan(user):
        if 'country' not in user:
            return False
        return user['country']['id'] == 4

    def _user_lda_by_groups(self, user):
        result = [0] * self.lda_maker.num_topics
        lda_count = 0
        for group in user['groups']:
            try:
                lda_desc = self.lda_maker.get(group['description'])
                for k, v in lda_desc:
                    result[k] += v
                lda_count += 1
            except KeyError:
                pass
        for i in range(len(result)):
            result[i] /= lda_count
        return result
