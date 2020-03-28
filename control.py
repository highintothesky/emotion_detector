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
from utils import class_indices
from tkinter import Tk, Frame, Button, Label, YES, X, Entry
from PIL import Image
from PIL import ImageTk
from queue import Queue
from time import sleep, time


class ControlWindow:

    def __init__(self, master, model_path, n_top=3):
        """Create the main window, then load the model."""
        master.title('OSC controller')
        self.master = master
        self.n_top = n_top
        self.sending = False
        self.remote_ip = '192.168.0.112'
        self.remote_port = 8000
        self.new_result = False
        # for populating the entry boxes
        self.initial_data = {'ip': self.remote_ip,
                             'port': self.remote_port}

        # self.label = Label(master, text="ports n shit")
        # self.label.pack()
        self.panel = None

        self.load_model(model_path)
        self.emo_queue = Queue()
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.video_capture = cv2.VideoCapture(0)
        self.stop_video_event = threading.Event()
        self.video_thread = threading.Thread(target=self.video_loop,
                                             args=(self.emo_queue,))
        self.video_thread.start()
        # OSC send threading event
        self.stop_send_event = threading.Event()

        self.init_inputs()
        # set a callback to handle when the window is closed
        self.master.wm_protocol("WM_DELETE_WINDOW", self.on_close)

    def init_inputs(self):
        """Initialize all the widgets for user input."""
        self.start_send_button = Button(self.master,
                                        text='Start OSC',
                                        command=self.start_send_osc)
        self.start_send_button.pack(side='bottom')

        side_frame = Frame(self.master)
        user_entries = []
        input_items = {1000: 'ip',
                       1001: 'port'}
        input_items.update(class_indices.copy())

        # emo can also be the ip or port
        for idx, emo in input_items.items():
            row = Frame(side_frame)
            lab = Label(row, width=15, text=emo, anchor='w')
            if emo in self.initial_data:
                init_text = self.initial_data[emo]
            else:
                init_text = ''
            ent = Entry(row)
            ent.insert(0, init_text)
            row.pack(side='top', fill=X, padx=5, pady=5)
            lab.pack(side='left')
            ent.pack(side='right', expand=YES, fill=X)
            user_entries.append((emo, ent))

        self.user_entries = user_entries
        side_frame.pack(side='right')

    def load_model(self, model_path):
        """Load emotion recognition model."""
        custom_objects = {'f1_m': f1_m,
                          'precision_m': precision_m,
                          'recall_m': recall_m}
        self.model = load_model(model_path,
                                custom_objects=custom_objects)

    def video_loop(self, emo_queue):
        """Loop over new frames."""
        try:
            last_time = time()
            # keep looping over frames until we are instructed to stop
            while not self.stop_video_event.is_set():
                # grab the frame from the video stream and resize it
                ret, self.frame = self.video_capture.read()
                image = cv2.resize(self.frame, (480, 360))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # if we're sending data, get the predictions and put them
                # on the queue
                if not self.stop_send_event.is_set():
                    self.top_classes = self.predict(image)
                    emo_queue.put(self.top_classes)
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
                new_time = time()
                loop_time = new_time - last_time
                last_time = new_time
                hz = 1./loop_time
                print(f'Running at {round(hz, 2)} Hz')

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
            entered_data = self.fetch_entries()
            # print('sending with data:')
            # print(entered_data)
            self.remote_ip = entered_data['ip']
            self.remote_port = int(entered_data['port'])
            self.sending = True
            self.stop_send_event = threading.Event()
            self.client = SimpleUDPClient(self.remote_ip, self.remote_port)
            self.osc_thread = threading.Thread(target=self.send_osc_loop,
                                               args=(self.emo_queue,))
            self.osc_thread.start()
        else:
            self.sending = False
            self.stop_send_event.set()

    def send_osc_loop(self, emo_queue):
        """Loop for constantly sending the class and score via OSC."""
        try:
            while not self.stop_send_event.is_set():
                emo = emo_queue.get()
                if len(emo) > 0:
                    print('got emo from queue:')
                    print(emo)
                    # print(type(emo))
                    self.client.send_message('emotion',
                                             emo[0]['emotion'])
                sleep(0.01)

        except RuntimeError as ex:
            print("[INFO] caught a RuntimeError while trying to send OSC")

    def fetch_entries(self):
        entered_data = {}
        for entry in self.user_entries:
            field = entry[0]
            text = entry[1].get()
            entered_data[field] = text
            print('%s: "%s"' % (field, text))
        return entered_data

    def on_close(self):
        """Clean up the threads and video capture."""
        print("[INFO] closing...")
        self.stop_video_event.set()
        if not self.stop_send_event.is_set():
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
