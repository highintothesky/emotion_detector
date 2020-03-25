# import os
import cv2
import click
import threading
# import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from train import recall_m, precision_m, f1_m
# from webcam import
from pythonosc.udp_client import SimpleUDPClient
from utils import preprocess_image, overlay_text, get_top_classes
from tkinter import Tk, Frame, Button, Label
from PIL import Image
from PIL import ImageTk


class ControlWindow:

    def __init__(self, master, model_path, n_top=3):
        """Create the main window, then load the model."""
        master.title("OSC controller")
        self.master = master
        self.n_top = n_top
        self.sending = False
        self.remote_ip = '192.168.0.112'
        self.remote_port = 8000
        self.new_result = False

        self.label = Label(master, text="ports n shit")
        self.label.pack()
        self.panel = None

        self.load_model(model_path)
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.video_capture = cv2.VideoCapture(0)
        self.stop_video_event = threading.Event()
        self.video_thread = threading.Thread(target=self.video_loop, args=())
        self.video_thread.start()

        self.start_send_button = Button(master,
                                        text="Start OSC",
                                        command=self.start_send_osc)
        self.start_send_button.pack(side="bottom")

        # set a callback to handle when the window is closed
        self.master.wm_protocol("WM_DELETE_WINDOW", self.on_close)

    def load_model(self, model_path):
        """Load emotion recognition model."""
        custom_objects = {'f1_m': f1_m,
                          'precision_m': precision_m,
                          'recall_m': recall_m}
        self.model = load_model(model_path,
                                custom_objects=custom_objects)

    def video_loop(self):
        """Loop over new frames."""
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stop_video_event.is_set():
                # grab the frame from the video stream and resize it
                ret, self.frame = self.video_capture.read()
                image = cv2.resize(self.frame, (480, 360))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.top_classes = self.predict(image)
                self.new_result = True
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="top", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
        except RuntimeError as ex:
            print("[INFO] caught a RuntimeError")

    def predict(self, rgb_img):
        """Scale down for network, get top classes."""
        rgb_img = rgb_img / 255.
        rgb_img = np.expand_dims(rgb_img, axis=0)
        res = self.model.predict(rgb_img)
        top_res = get_top_classes(res, n=self.n_top)
        return top_res

    def start_send_osc(self):
        if not self.sending:
            self.sending = True
            self.stop_send_event = threading.Event()
            self.client = SimpleUDPClient(self.remote_ip, self.remote_port)
            self.osc_thread = threading.Thread(target=self.send_osc_loop,
                                               args=())
            self.osc_thread.start()

    def send_osc_loop(self):
        """Loop for constantly sending the class and score via OSC."""
        try:
            while not self.stop_send_event.is_set():
                if self.new_result:
                    print(self.top_classes[0])
                    self.client.send_message('emotion',
                                             self.top_classes[0]['emotion'])
                    self.new_result = False
        except RuntimeError as ex:
            print("[INFO] caught a RuntimeError while trying to send OSC")

    def on_close(self):
        """Clean up the threads and video capture."""
        print("[INFO] closing...")
        self.stop_video_event.set()
        self.stop_send_event.set()
        self.video_capture.release()
        self.master.quit()


@click.command()
@click.option('--model',
              default='models/best_model.h5',
              help='Path to your data folder.')
def main(**kwargs):
    """Start the main Tkinter window."""
    root = Tk()
    gui = ControlWindow(root, kwargs['model'])
    root.mainloop()


if __name__ == '__main__':
    main()
