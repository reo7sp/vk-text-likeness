import os

from vk_text_likeness.predict_main import GroupPredict

if __name__ == '__main__':
    group_id = int(os.sys.argv[1])
    assert group_id > 0
    access_token = os.sys.argv[2]

    group_predict = GroupPredict(group_id, access_token)

    predictions = group_predict.predict()
    with open('predictions.csv', 'w') as f:
        f.write(predictions.to_csv())

    true = group_predict.get_true(predictions.index)
    with open('true.csv', 'w') as f:
        f.write(true.to_csv())
