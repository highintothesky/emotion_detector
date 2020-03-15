"""Webcam emotion recognition network test. Also doubles as data recorder."""
# import tensorflow as tf
import os
import cv2
import click
import pandas as pd


def overlay_text(img, emotion, state, recording, bottomLeftCornerOfText,
                 font, fontScale, fontColor, lineType):
    """Overlay status text on image."""
    if state == 'recording' and recording:
        img = cv2.circle(img, (30, 30), 10, (0, 0, 255), -1)
    elif state == 'recording' and not recording:
        img = cv2.circle(img, (30, 30), 10, (255, 255, 0), -1)
    cv2.putText(
        img,
        emotion,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType
    )
    return img


@click.command()
@click.option('--data_path',
              default='data',
              help='Path to your data folder.')
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
    bottomLeftCornerOfText = (500, 450)
    fontScale = 0.7
    fontColor = (0, 255, 0)  # BGR because of opencv
    lineType = 2

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

        im2show = frame.copy()
        # display current emotion selected
        im2show = overlay_text(im2show, emotions[current_emotion], state,
                               recording, bottomLeftCornerOfText, font,
                               fontScale, fontColor, lineType)
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
        # elif key & 0xFF == ord('1'):
        #     print(1111)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
