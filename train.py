"""Main training script."""
import click
import tensorflow as tf
import pandas as pd
import numpy as np


@click.command()
@click.option('--data_path',
              default='data',
              help='Path to your data folder.')
@click.option('--model_name',
              default='new_model.h5',
              help='Name of your new model.')
def main(**kwargs):
    

if __name__ == '__main__':
    main()
