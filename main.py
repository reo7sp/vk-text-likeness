import os

from vk_text_likeness.predict_main import GroupPredict

if __name__ == '__main__':
    group_id = int(os.sys.argv[1])
    assert group_id > 0
    access_token = os.sys.argv[2]

    group_predict = GroupPredict(group_id, access_token)
    predictions = group_predict.predict()
    check = group_predict.check(predictions)

    print()
    print('-' * 80)
    print(check.to_csv())
    try:
        with open('result.csv', 'w') as f:
            f.write(predictions.to_csv())
    except IOError as e:
        print(e)
        pass
