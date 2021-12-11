import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
                self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):

        # your_code
        user_item_matrix = pd.pivot_table(data_train,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype('float')

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        self.model = AlternatingLeastSquares(factors=factors,
                                             regularization=regularization,
                                             iterations=iterations,
                                             num_threads=num_threads)
        self.model.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return self.model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре

        res = []
        
        top_N_items = user_item_matrix.loc[user].sort_values(ascending=False)[:N].copy()
        top_N_items.index = [self.itemid_to_id[ind] for ind in top_N_items.index]

        for i, item in enumerate(top_N_items):
            recs = self.model.similar_items(item, N=N + i + 1)
            recs = [rec[0] for rec in recs if rec[0] not in aaa.index and rec[0] not in res]
            res.append(self.id_to_itemid[recs[0]])

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code
        sim_users = model.similar_users(user, N=N + 1)[1:]
        sim_users = [self.userid_to_id[user[0]] for user in sim_users]
        items = self.user_item_matrix.loc[sim_users].copy()
        items.columns = [self.itemid_to_id[clm] for clm in items.columns]
        items = items.T
        items = items.index[items.any(axis=1)]

        res = pd.Series(self.model.item_factors[items] @ self.model.user_factors[self.userid_to_id[user]], index = items, name='score')
        res = [self.id_to_itemid[rec] for rec in res.sort_values(ascending=False)[:5].index]


        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res