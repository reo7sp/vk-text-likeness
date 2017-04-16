import vk_api

from vk_text_likeness.lda_maker import LdaMaker
from vk_text_likeness.logs import log_method_begin, log_method_end
from vk_text_likeness.tools import cache_by_entity_id


class RawWallData:
    def __init__(self, group_id, vk_session):
        self.group_id = group_id
        self.vk_session = vk_session
        self.vk = self.vk_session.get_api()
        self.vk_tools = vk_api.VkTools(self.vk_session)

        self.posts = []

    def fetch(self):
        self._fetch_wall()
        self._fetch_activity()

    def _fetch_wall(self):
        log_method_begin()
        self.posts = self.vk_tools.get_all('wall.get', 100, {'owner_id': -self.group_id, 'extended': 1})['items']
        print('{} posts'.format(len(self.posts)))
        log_method_end()

    def _fetch_activity(self):
        log_method_begin()

        pool_results = []

        with vk_api.VkRequestsPool(self.vk_session) as pool:
            for post in self.posts:
                likes = pool.method(
                    'likes.getList',
                    {'item_id': post['id'], 'owner_id': -self.group_id, 'type': 'post', 'count': 1000, 'filter': 'likes'}
                )
                reposts = pool.method(
                    'likes.getList',
                    {'item_id': post['id'], 'owner_id': -self.group_id, 'type': 'post', 'count': 1000, 'filter': 'copies'}
                )
                pool_results.append((post, likes, reposts))

        for post, likes, reposts in pool_results:
            if 'likes' not in post:
                post['likes'] = dict()
            if likes.ok:
                likes = likes.result['items']
                post['likes']['user_ids'] = set(likes)
            else:
                post['likes']['user_ids'] = set()

            if 'reposts' not in post:
                post['reposts'] = dict()
            if reposts.ok:
                reposts = reposts.result['items']
                post['reposts']['user_ids'] = set(reposts)
            else:
                post['reposts']['user_ids'] = set()

        log_method_end()


class TableWallData:
    def __init__(self, raw_wall_data, num_topics=15):
        self.raw_wall_data = raw_wall_data
        self.num_topics = num_topics

    def fit(self):
        log_method_begin()
        self.lda_maker = LdaMaker(self._get_corpora_for_lda(), self.num_topics)
        log_method_end()

    @cache_by_entity_id
    def get_row(self, post):
        return [self._post_text_len(post)] + self._post_lda(post)

    def get_labels(self):
        return ['text_len'] + ['post_lda' + str(i) for i in range(self.lda_maker.num_topics)]

    def _get_corpora_for_lda(self):
        corpora = []
        for post in self.raw_wall_data.posts:
            doc = post['text']
            corpora.append(doc)
        return corpora

    @staticmethod
    def _post_text_len(post):
        return len(post['text'])

    def _post_lda(self, post):
        result = [0] * self.lda_maker.num_topics
        lda_text = self.lda_maker.get(post['text'])
        for k, v in lda_text:
            result[k] = v
        return result
