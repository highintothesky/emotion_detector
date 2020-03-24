"""Common utility functions for emotion detection."""
import cv2
import numpy as np


class_indices = {0: 'angry',
                 1: 'confused',
                 2: 'crosseyed',
                 3: 'happy',
                 4: 'neutral',
                 5: 'sad'}


def preprocess_image(image):
    """Scale, reshape etc."""
    image = cv2.resize(image, (480, 360))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    return image


def overlay_text(img, emotion, state, recording, bottomLeftCornerOfText,
                 font, fontScale, fontColor, lineType, row_dy):
    """Overlay status text on image."""
    if state == 'recording' and recording:
        img = cv2.circle(img, (30, 30), 10, (0, 0, 255), -1)
    elif state == 'recording' and not recording:
        img = cv2.circle(img, (30, 30), 10, (255, 255, 0), -1)

    x = bottomLeftCornerOfText[0]
    y = bottomLeftCornerOfText[1]
    rows = emotion.split('\n')
    y_pos = [y - i*row_dy for i in range(len(rows))]
    y_pos.reverse()

    for i, line in enumerate(rows):
        y = y_pos[i]
        textpos = (x, y)
        cv2.putText(
            img,
            line,
            textpos,
            font,
            fontScale,
            fontColor,
            lineType
        )
    return img


def get_top_classes(network_output, n=2):
    """Convert array to n human readable classes and scores."""
    idx = (-network_output[0, :]).argsort()[:n]
    scores = []
    for i in range(n):
        score = network_output[0, idx[i]]
        emotion = class_indices[idx[i]]
        scores.append({'emotion': emotion, 'score': score})
    # emo1 = str(round(score1*100, 2)) + '% ' + class_indices[idx[0]]
    # score2 = network_output[0, idx[1]]
    # emo2 = str(round(score2*100, 2)) + '% ' + class_indices[idx[1]]
    return scores
