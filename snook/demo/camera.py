import cv2
import numpy as np
import threading

from typing import Tuple


class Camera:
    def __init__(self, src: int, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(src)
        self.width, self.height = width, height
        self.started = False
        self.read_lock = threading.Lock()
        self.grabbed, self.frame = self.cap.read()

    @property
    def width(self) -> int:
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    @width.setter
    def width(self, value: int) -> None:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, value)

    @property
    def height(self) -> int:
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @height.setter
    def height(self, value: int) -> None:
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value)

    def start(self) -> "Camera":
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target = self.update, args = ())
        self.thread.start()
        return self

    def update(self) -> None:
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self) -> Tuple[bool, np.ndarray]:
        with self.read_lock:
            frame = self.frame.copy()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        return self.grabbed, np.array(frame) / 255.0

    def stop(self) -> None:
        self.started = False
        self.cap.release()
        self.thread.join()