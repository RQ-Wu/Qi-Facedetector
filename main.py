import cv2
import time
import threading
from cv2 import cv2
from PIL import Image, ImageTk
from tkinter import Label, Button, Tk, PhotoImage
from face_detect import face_detector, liveness_detector, ID_recognize
from util import *
import numpy as np
import matplotlib.pyplot as plt

class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("人脸签到系统")
        self.window.geometry("640x560")
        self.window.configure(bg="#ff2fff")
        self.window.resizable(1, 1)
        Label(self.window, width=400, height=30, bg="black").place(x=0, y=320)
        self.TakePhoto_b = Button(self.window, width=20, text="签到", font=("Times", 10), bg="#2F4F4F", relief='flat',
                                  command=self.TakePhoto)
        self.ImageLabel = Label(self.window, width=640, height=480, bg="#4682B4")
        self.ImageLabel.place(x=0, y=0)
        self.TakePhoto_b.place(x=250, y=520)
        self.i = 0
        self.frames = ['', '', '', '', '']
        self.take_picture = False
        self.PictureTaken = False
        self.Main()

    @staticmethod
    def LoadCamera():
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            ret, frame = camera.read()
        while ret:
            ret, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                yield frame
            else:
                yield False

    def TakePhoto(self):
        self.take_picture = True

    def Main(self):
        self.render_thread = threading.Thread(target=self.StartCamera)
        self.render_thread.daemon = True
        self.render_thread.start()

    def StartCamera(self):
        frame = self.LoadCamera()
        CaptureFrame = None
        while True:
            Frame = next(frame)
            if frame and not self.take_picture:
                self.frames[self.i] = Frame
                self.i = (self.i + 1) % 5
                detector = face_detector(Frame)
                picture = Image.fromarray(detector)
                picture = ImageTk.PhotoImage(picture)
                self.ImageLabel.configure(image=picture)
                self.ImageLabel.photo = picture
                self.PictureTaken = False
                time.sleep(0.001)
            else:
                live_flag = True
                # live detector
                for capture in self.frames:
                    if not liveness_detector(capture):
                        self.window.title("签到失败！未检查到人脸")
                        live_flag = False
                        break
                if live_flag:
                    prob_avg = 0
                    # ID recognize
                    embedding_dict = load_embedding()
                    for capture in self.frames:
                        index, prob = ID_recognize(capture)
                        print(embedding_dict[index]['name'], prob)
                        prob_avg = prob_avg + prob
                    if prob_avg / 5.0 < 0.95:
                        self.window.title(f"{embedding_dict[index]['name']}  签到成功！")
                        save_check(embedding_dict[index]['name'])
                    else:
                        self.window.title("签到失败！未查询到身份")
                self.take_picture = False


root = Tk()
App = CameraApp(root)
root.mainloop()