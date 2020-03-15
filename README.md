# emotion_detector
Face emotion detection using tf.keras

## Preparation

start a virtualenv:

`python3 -m venv .venv`

activate it:

`source .venv/bin/activate`

update your pip:

`pip3 install --upgrade pip`

install dependencies:

`pip3 install -r requirements.txt`

now, If you have a GPU and CUDA 10.2 installed:

`pip3 install tensorflow-gpu==2.0.1`

else:

`pip3 install tensorflow-cpu==2.1.0`

## Testing/Data Recording

start the webcam demo:

`python3 webcam.py`

in the webcam GUI, press 't' to toggle between data recording and testing,
0-5 to select the target emotion, 'r' to start recording, and 'q' to stop
and write the data to your data folder.
