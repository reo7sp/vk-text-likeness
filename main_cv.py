import os

from sklearn.model_selection import KFold

from vk_text_likeness.check import check
from vk_text_likeness.predict_main import GroupPredict

if __name__ == '__main__':
    group_id = int(os.sys.argv[1])
    assert group_id > 0
    access_token = os.sys.argv[2]

    group_predict = GroupPredict(group_id, access_token)
    group_predict.prepare()

    cv = KFold(5, shuffle=True)
    i = 0
    for train_index, test_index in cv.split(group_predict.action_data.get_all()):
        print('\nCV: iter #{}'.format(i))
        group_predict.fit(train_index)
        predictions_df = group_predict.predict(test_index)
        true_df = group_predict.get_true(predictions_df.index)
        check(predictions_df, true_df)
        i += 1
