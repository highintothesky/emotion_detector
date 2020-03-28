"""Webcam emotion recognition network test. Also doubles as data recorder."""
import os
import cv2
import click
import pandas as pd
# import numpy as np
from tensorflow.keras.models import load_model
from train import recall_m, precision_m, f1_m
from utils import preprocess_image, overlay_text, get_top_classes


@click.command()
@click.option('--data_path',
              default='data',
              help='Path to your data folder.')
@click.option('--model',
              default='models/best_model.h5',
              help='Path to your model file.')
@click.option('--n_top',
              default=3,
              help='How many of the resulting emotions to show.')
def main(**kwargs):
    """Load model, start webcam, run loop."""
    video_capture = cv2.VideoCapture(0)
    # variables for registering emotion recording
    emotions = ['neutral',
                'happy',
                'sad',
                'angry',
                'confused',
                'crosseyed']
    allowed_idx = list(range(len(emotions)))
    current_emotion = 0
    state = 'recording'  # recording | testing (network output)
    recording = False
    # some settings for printing on screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (400, 450)
    fontScale = 0.7
    fontColor = (0, 255, 0)  # BGR because of opencv
    lineType = 2
    row_dy = 20
    # load the model with custom functions
    class_indices = {0: 'angry',
                     1: 'confused',
                     2: 'crosseyed',
                     3: 'happy',
                     4: 'neutral',
                     5: 'sad'}
    custom_objects = {'f1_m': f1_m,
                      'precision_m': precision_m,
                      'recall_m': recall_m}
    model = load_model(kwargs['model'],
                       custom_objects=custom_objects)

    # load previous data so we can add to it
    csv_path = os.path.join(kwargs['data_path'], 'all.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame(columns=['path', 'emotion'])

    while True:
        ret, frame = video_capture.read()

        # record it if the user tells us to
        if state == 'recording' and recording:
            img_idx = df.index.max()
            if img_idx != img_idx:
                img_idx = 0
            else:
                img_idx += 1
            img_name = f'{emotions[current_emotion]}_{img_idx}.png'
            img_path = os.path.join(kwargs['data_path'], 'images', img_name)
            cv2.imwrite(img_path, frame)
            df = df.append(
                {'path': img_path, 'emotion': emotions[current_emotion]},
                ignore_index=True
            )

        emotion = emotions[current_emotion]
        if state == 'testing':
            # get the network output
            img = preprocess_image(frame)
            res = model.predict(img)
            top_res = get_top_classes(res, n=kwargs['n_top'])
            emotion = ''
            for sc_dict in top_res:
                emotion += str(round(sc_dict['score']*100, 2)) + '% '
                emotion += sc_dict['emotion'] + '\n'

        im2show = frame.copy()
        # display current emotion selected
        im2show = overlay_text(im2show, emotion, state,
                               recording, bottomLeftCornerOfText, font,
                               fontScale, fontColor, lineType, row_dy)
        cv2.imshow('Video', im2show)

        # control handling
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            df.to_csv(csv_path)
            break
        elif key & 0xFF == ord('t'):
            if state == 'recording':
                state = 'testing'
            else:
                state = 'recording'
            print('Setting state to', state)
        elif key & 0xFF == ord('r'):
            if recording is True:
                recording = False
            else:
                recording = True
            print('Setting recording to', recording)
        # set emotion to record
        for idx in allowed_idx:
            if key & 0xFF == ord(str(idx)):
                print('Setting emotion recording to', emotions[idx])
                current_emotion = idx

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
