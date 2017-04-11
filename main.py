import click

from vk_text_likeness.predict import GroupPredict

@click.command()
@click.argument('group_id')
@click.argument('access_token')
def predict(group_id, access_token):
    group_predict = GroupPredict(group_id, access_token)
    df = group_predict.predict()
    print()
    print('-' * 80)
    print(df.to_csv())

if __name__ == '__main__':
    predict()
