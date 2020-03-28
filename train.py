"""Main training script."""
import os
import click
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def build_model():
    """Build a simple CNN."""
    input_layer = Input(shape=(None, None, 3))
    x = Conv2D(32, (3, 3), activation='selu')(input_layer)
    # x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='selu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (4, 4), activation='selu')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(128, (1, 1))(x)

    # this allows for pooling without losing image dimension flexibility
    x = GlobalMaxPooling2D()(x)

    # some fully connected on top
    x = Dense(64, activation='selu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(6, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


@click.command()
@click.option('--data_path',
              default='data',
              help='Path to your data folder.')
@click.option('--model_name',
              default='new_model.h5',
              help='Name of your new model.')
@click.option('--batch_size',
              default=16,
              help='Training batch size.')
def main(**kwargs):
    """make a new model, train it."""
    train_csv = os.path.join(kwargs['data_path'], 'train', 'data.csv')
    test_csv = os.path.join(kwargs['data_path'], 'test', 'data.csv')
    train_img_path = os.path.join(kwargs['data_path'], 'train', 'images')
    test_img_path = os.path.join(kwargs['data_path'], 'test', 'images')
    model_path = os.path.join('models', kwargs['model_name'])

    train_df = pd.read_csv(train_csv, index_col=0)
    valid_df = pd.read_csv(test_csv, index_col=0)

    model = build_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    print(model.summary())
    print(train_df)
    print(valid_df)

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=30,
            horizontal_flip=True,
            fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_img_path,
        x_col='name',
        y_col='emotion',
        target_size=(360, 480),
        batch_size=kwargs['batch_size'],
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=test_img_path,
        x_col='name',
        y_col='emotion',
        target_size=(360, 480),
        batch_size=kwargs['batch_size'],
        class_mode='categorical')

    print('class indices:')
    print(validation_generator.class_indices)

    monitor = 'val_accuracy'
    check_callback = ModelCheckpoint(
        model_path,
        monitor=monitor,
        save_best_only=True)

    check_early = EarlyStopping(
        monitor=monitor,
        patience=15,
        verbose=1)

    check_tb = TensorBoard(
        log_dir='logs',
        write_images=True,
        histogram_freq=2,
        write_graph=True,
        embeddings_freq=2
    )

    model.fit_generator(
        train_generator,
        epochs=80,
        validation_data=validation_generator,
        callbacks=[check_callback, check_early, check_tb])



if __name__ == '__main__':
    main()
