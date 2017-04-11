import os

from vk_text_likeness.predict import GroupPredict

if __name__ == '__main__':
    group_id = int(os.sys.argv[1])
    assert group_id > 0
    access_token = os.sys.argv[2]

    group_predict = GroupPredict(group_id, access_token)
    df = group_predict.predict()
    print()
    print('-' * 80)
    print(df.to_csv())
