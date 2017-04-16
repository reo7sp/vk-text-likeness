from collections import defaultdict
from datetime import date

import vk_api

from vk_text_likeness.lda_maker import LdaMaker
from vk_text_likeness.tools import cache_by_entity_id
from vk_text_likeness.logs import log_method_begin, log_method_end


class RawUsersData:
    def __init__(self, group_id, vk_session):
        self.group_id = group_id
        self.vk_session = vk_session
        self.vk = self.vk_session.get_api()
        self.vk_tools = vk_api.VkTools(self.vk_session)

        self.members = []
        self.member_friends = defaultdict(list)

        self.member_fields = 'sex,bdate,country'
        self.group_fields = 'description'

    def fetch(self):
        self._fetch_members()
        self._fetch_member_friends()
        self._fetch_groups()

    def _fetch_members(self):
        log_method_begin()

        self.members = self.vk_tools.get_all(
            'groups.getMembers', 1000, {'group_id': self.group_id, 'fields': self.member_fields}
        )['items']
        print('{} members'.format(len(self.members)))

        for member in self.members:
            member['is_member'] = True

        log_method_end()

    def _fetch_member_friends(self):
        log_method_begin()

        pool_results = []

        with vk_api.VkRequestsPool(self.vk_session) as pool:
            for member in self.members:
                pool_results.append(
                    (member['id'], pool.method('friends.get', {'user_id': member['id'], 'fields': 'photo'}))
                )

        self.member_friends = defaultdict(list)
        member_ids = set(user['id'] for user in self.members)
        for member_id, friend_request in pool_results:
            if friend_request.ok:
                for friend in friend_request.result['items']:
                    if friend['id'] not in member_ids:
                        friend['is_member'] = False
                        self.member_friends[member_id].append(friend)

        log_method_end()

    def _fetch_groups(self):
        log_method_begin()

        users = self.get_all_users()
        print('{} users to fetch'.format(len(users)))

        with vk_api.VkRequestsPool(self.vk_session) as pool:
            for user in users:
                user['groups'] = pool.method('groups.get', {
                    'user_id': user['id'], 'count': 1000, 'extended': 1, 'fields': self.group_fields
                })

        for user in users:
            if user['groups'].ok:
                user['groups'] = user['groups'].result['items']
            else:
                user['groups'] = []

        log_method_end()

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
        log_method_begin()
        self.lda_maker = LdaMaker(self._get_corpora_for_lda(), self.num_topics)
        log_method_end()

    @cache_by_entity_id
    def get_row(self, user):
        return [self._user_is_woman(user), self._user_is_man(user), self._user_age(user),
                self._user_is_in_russia(user), self._user_is_in_ukraine(user), self._user_is_in_byelorussia(user),
                self._user_is_in_kazakstan(user)] + \
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
