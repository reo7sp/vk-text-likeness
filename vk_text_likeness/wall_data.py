import pandas as pd
import vk_api
from tqdm import tqdm

from vk_text_likeness.lda_maker import LdaMaker
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
        self.posts = self.vk_tools.get_all('wall.get', 100, {'owner_id': -self.group_id, 'extended': 1})['items']

    def _fetch_activity(self):
        for post in tqdm(self.posts, 'RawWallData._fetch_activity: for posts'):
            likes = self.vk.likes.getList(filter='likes', item_id=post['id'], owner_id=-post['owner_id'], count=1000,
                                          **{'type': 'post'})
            likes = likes['items']
            post['likes'] = dict() if 'likes' not in post else post['likes']
            post['likes']['users'] = likes

            reposts = self.vk.likes.getList(filter='copies', item_id=post['id'], owner_id=-post['owner_id'], count=1000,
                                            **{'type': 'post'})
            reposts = reposts['items']
            post['reposts'] = dict() if 'likes' not in post else post['reposts']
            post['reposts']['users'] = reposts


class TableWallData:
    def __init__(self, raw_wall_data, num_topics=15):
        self.raw_wall_data = raw_wall_data
        self.num_topics = num_topics

    def fit(self):
        self.lda_maker = LdaMaker(self._get_corpora_for_lda(), self.num_topics)

    def get_all(self):
        posts = self.raw_wall_data.posts
        return pd.DataFrame([self.get_row(post) for post in tqdm(posts, 'TableWallData.get_all: for posts')], index=[post['id'] for post in posts])

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
