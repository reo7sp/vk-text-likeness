import os
import random
import traceback
from collections import defaultdict
from datetime import date
import time

import pickle
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

        self.members = None
        self.member_friends = None

        self.member_fields = 'sex,bdate,country'
        self.group_fields = 'description'

        self.fetch_groups_mark_file = 'RawUsersData._fetch_groups_error{}'.format(self.group_id)

    def fetch(self):
        self._fetch_members()

    def fetch_more(self, liked_users_set, reposted_users_set):
        self._fetch_member_friends(reposted_users_set)
        self._fetch_groups(liked_users_set | self._sample_user_ids(len(liked_users_set), without=liked_users_set))

    def _fetch_members(self):
        if self.members is not None:
            return
        log_method_begin()

        self.members = self.vk_tools.get_all(
            'groups.getMembers', 1000, {'group_id': self.group_id, 'fields': self.member_fields}
        )['items']
        print('{} members'.format(len(self.members)))

        for member in self.members:
            member['is_member'] = True

        log_method_end()

    def _fetch_member_friends(self, user_subset):
        if self.member_friends is not None:
            return
        log_method_begin()

        members = [member for member in self.members if member['id'] in user_subset]
        print('{} users to fetch'.format(len(members)))

        pool_results = []

        with vk_api.VkRequestsPool(self.vk_session) as pool:
            for member in members:
                pool_results.append(
                    (member['id'], pool.method('friends.get', {'user_id': member['id'], 'fields': 'photo'}))
                )

        self.member_friends = defaultdict(list)
        for member_id, friend_request in pool_results:
            if friend_request.ok:
                for friend in friend_request.result['items']:
                    if friend['id'] not in user_subset:
                        friend['is_member'] = False
                        self.member_friends[member_id].append(friend)

        log_method_end()

    def _fetch_groups(self, user_subset):
        log_method_begin()

        all_users = [user for user in self.get_all_users() if user['id'] in user_subset]
        print('{} users to fetch'.format(len(all_users)))

        all_users_processing_step = 1000
        fetch_start = time.time()
        for i in range(0, len(all_users), all_users_processing_step):
            print('Fetching from {} to {}...'.format(i, i + all_users_processing_step))
            users = all_users[i:i+all_users_processing_step]

            if time.time() - fetch_start > 2 * 60 * 60:
                print('Cooldown for 30 minutes')
                time.sleep(30 * 60)
                fetch_start = time.time()

            do_fetch = True
            last_error_time = -1
            while do_fetch:
                try:
                    pool_results = []
                    with vk_api.VkRequestsPool(self.vk_session) as pool:
                        for user in users:
                            if 'groups' not in user:
                                pool_results.append(
                                    (user, pool.method('groups.get', {'user_id': user['id'], 'count': 1000, 'extended': 1, 'fields': self.group_fields}))
                                )
                    do_fetch = False
                    self.unmark_fetch_groups_error()
                except Exception as e:
                    print('Can\'t fetch groups because of', e)
                    traceback.print_exc()
                    if time.time() - last_error_time < 120:
                        print('Can\'t do anything, exit. Restart will reuse fetched users')
                        do_fetch = False
                        self.mark_fetch_groups_error()
                    else:
                        print('Trying again in 1 minute')
                        time.sleep(60)
                    last_error_time = time.time()
                finally:
                    for user, groups_request in pool_results:
                        if groups_request.ok and groups_request.ready:
                            user['groups'] = [{'description': group['description']}
                                              for group in groups_request.result['items']]

            self._save_pickle('raw_users_data.members', self.members)
            self._save_pickle('raw_users_data.member_friends', self.member_friends)

        log_method_end()

    def was_fetch_groups_error(self):
        return os.path.isfile(self.fetch_groups_mark_file)

    def mark_fetch_groups_error(self):
        with open(self.fetch_groups_mark_file, 'w') as f:
            f.write('True')

    def unmark_fetch_groups_error(self):
        if self.was_fetch_groups_error():
            os.remove(self.fetch_groups_mark_file)

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
        everybody = []
        ids = set()
        for users in [self.members] + list(self.member_friends.values()):
            for user in users:
                if user['id'] not in ids:
                    everybody.append(user)
                    ids.add(user['id'])
        return everybody

    def _sample_user_ids(self, n, without=set()):
        ids = set()
        for users in [self.members] + list(self.member_friends.values()):
            for user in users:
                if user['id'] not in ids and user['id'] not in without:
                    ids.add(user['id'])
        return set(random.sample(ids, n))

    def _save_pickle(self, name, obj):
        try:
            with open('{}{}.pkl'.format(name, self.group_id), 'wb') as f:
                pickle.dump(obj, f)
        except IOError as e:
            print('Can\'t save pickle {}:'.format(name), e)


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
            if 'groups' in user:
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
        if 'groups' in user:
            for group in user['groups']:
                try:
                    lda_desc = self.lda_maker.get(group['description'])
                    for k, v in lda_desc:
                        result[k] += v
                    lda_count += 1
                except KeyError:
                    pass
            if lda_count != 0:
                for i in range(len(result)):
                    result[i] /= lda_count
        return result
