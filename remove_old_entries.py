"""Script for removing old images from csv."""
import click
import os
import pandas as pd


@click.command()
@click.option('--csv_path',
              default='data/all.csv',
              help='Path to your data folder.')
def main(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    print('length before:', len(df))
    for idx, row in df.iterrows():
        if not os.path.isfile(row['path']):
            df.drop(index=idx, inplace=True)
    print('length after:', len(df))
    df.to_csv(csv_path)


if __name__ == '__main__':
    main()
