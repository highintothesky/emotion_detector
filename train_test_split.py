"""Script for splitting data into train and test set."""
import os
import glob
import click
import random
import ntpath
import pandas as pd
from shutil import copyfile


@click.command()
@click.option('--data_path',
              default='./data',
              help='Path to your data folder.')
@click.option('--train_ratio',
              default=0.75,
              help='Part of the data to use as training set.')
def main(**kwargs):
    csv_path = os.path.join(kwargs['data_path'], 'all.csv')
    train_path = os.path.join(kwargs['data_path'], 'train')
    test_path = os.path.join(kwargs['data_path'], 'test')
    train_img_path = os.path.join(train_path, 'images')
    test_img_path = os.path.join(test_path, 'images')
    train_df_path = os.path.join(train_path, 'data.csv')
    test_df_path = os.path.join(test_path, 'data.csv')

    df = pd.read_csv(csv_path, index_col=0)
    df_train = pd.DataFrame(columns=['path', 'name', 'emotion'])
    df_test = pd.DataFrame(columns=['path', 'name', 'emotion'])

    # remove the old files
    for path in [train_img_path, test_img_path]:
        _str = os.path.join(path, '*')
        files = glob.glob(_str)
        for file in files:
            os.remove(file)

    for idx, row in df.iterrows():
        src = row['path']
        name = ntpath.basename(src)
        if random.random() < kwargs['train_ratio']:
            dst = os.path.join(train_img_path, name)
            df_train = df_train.append(
                {'path': dst, 'name': name, 'emotion': row['emotion']},
                ignore_index=True
            )
        else:
            dst = os.path.join(test_img_path, name)
            df_test = df_test.append(
                {'path': dst, 'name': name, 'emotion': row['emotion']},
                ignore_index=True
            )
        copyfile(src, dst)

    df_train.to_csv(train_df_path)
    df_test.to_csv(test_df_path)



if __name__ == '__main__':
    main()
