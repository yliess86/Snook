import cv2
import numpy as np
import threading
import torch
import torch.nn.functional as F

from glumpy import app, gl, gloo
from typing import Tuple


BLACK = (0.2, 0.2, 0.2, 0.5)
WHITE = (0.8, 0.8, 0.8, 0.5)
RED = (0.8, 0.1, 0.2, 0.5)
YELLOW = (0.6, 0.6, 0.2, 0.5)
CUE = (0.1, 0.2, 0.8, 0.5)
LABEL2COLOR = { label: color for label, color in enumerate([BLACK, WHITE, YELLOW, RED, CUE]) }

QUAD_UV = [(0, 1), (0, 0), (1, 1), (1, 0)]
VS = """
attribute vec2 pos; attribute vec2 uv; varying vec2 v_uv;
void main() { gl_Position = vec4(pos, 0., 1.); v_uv = uv; }
"""
FS = """
uniform sampler2D texture; varying vec2 v_uv;
void main() { gl_FragColor = texture2D(texture, v_uv).rgba; }
"""

CAMERA, CAMERA_W, CAMERA_H = 2, 960, 540
INPUT_W, INPUT_H = 256, 256
APP_W, APP_H = INPUT_W * 2 * 3, INPUT_H * 2
THRESHOLD = 0.2


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

    def read(self, width: int, height: int) -> Tuple[bool, np.ndarray]:
        with self.read_lock:
            frame = self.frame.copy()

        factor = np.max([width, height]) / np.max([self.width, self.height])
        w, h = int(factor * self.width), int(factor * self.height)
        
        frame = cv2.resize(frame, (w, h))
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        
        offset_w, offset_h = (width - w) // 2, (height - h) // 2
        result = np.zeros((height, width, 3))
        result[offset_h:offset_h + h, offset_w:offset_w + w] = frame

        return self.grabbed, np.array(result) / 255.0

    def stop(self) -> None:
        self.started = False
        self.cap.release()
        self.thread.join()


def heatmap2loc(x: torch.Tensor, threshold: float) -> torch.Tensor:
    W, H = x.size()

    x = torch.clamp(x, 0, 1)
    x[x < threshold] = 0

    border = F.pad(x, [1] * 4, mode="constant")
    center = border[1:1 + W, 1:1 + H]
    left   = border[1:1 + W, 2:2 + H]
    right  = border[1:1 + W, 0:0 + H]
    up     = border[2:2 + W, 1:1 + H]
    down   = border[0:0 + W, 1:1 + H]
    
    peaks  = (center > left) & (center > right) & (center > up) & (center > down)
    coords = torch.nonzero(peaks, as_tuple=False)
    for pos in coords:
        pos = pos.unsqueeze(0)
        dist = torch.sqrt(torch.sum((coords.float() - pos) ** 2, dim=1))
        reject = (dist < 2) & (dist > 1e-3)
        coords = coords[~reject]

    return coords


def loc2window(render: torch.Tensor, locations: torch.Tensor, size: int = 32) -> torch.Tensor:
    *_, w, h = render.size()
    offset = size // 2
    samples = torch.zeros((locations.size(0), 3, size, size)).float().cuda()
    for i, location in enumerate(locations):
        x_start, x_end = np.min([np.max([location[0] - offset, 0]), w]), np.min([np.max([location[0] + offset, 0]), w])
        y_start, y_end = np.min([np.max([location[1] - offset, 0]), h]), np.min([np.max([location[1] + offset, 0]), h])
        x_span, y_span = x_end - x_start, y_end - y_start
        samples[i, :, :x_span, :y_span] = render[:, :, x_start:x_end, y_start:y_end]
    return samples


locnet   = torch.jit.load("./resources/model/locnet.ts"  ).eval().cuda()
masknet  = torch.jit.load("./resources/model/masknet.ts" ).eval().cuda()
labelnet = torch.jit.load("./resources/model/labelnet.ts").eval().cuda()

camera = Camera(CAMERA, CAMERA_W, CAMERA_H)

window = app.Window(width=APP_W, height=APP_H, title="Snook", fullscreen=False)
quad_v = gloo.Program(VS, FS, count=4)
quad_m = gloo.Program(VS, FS, count=4)
quad_h = gloo.Program(VS, FS, count=4)


@window.event
def on_init():
    frac = 2 / 3
    quad_v["pos"] = [( -1       , -1), (-1 + frac, -1), (-1       , 1), (-1 + frac, 1)][::-1]
    quad_m["pos"] = [( -1 + frac, -1), ( 1 - frac, -1), (-1 + frac, 1), ( 1 - frac, 1)][::-1]
    quad_h["pos"] = [(  1 - frac, -1), ( 1       , -1), ( 1 - frac, 1), ( 1       , 1)][::-1]
    quad_v["uv"] = quad_m["uv"] = quad_h["uv"] = QUAD_UV
    camera.start()


@window.event
def on_draw(dt):
    _, capture = camera.read(INPUT_W, INPUT_H)

    with torch.no_grad():
        render  = torch.from_numpy(capture).float().permute((2, 0, 1)).unsqueeze(0).cuda()
        mask    = torch.clamp(masknet(render), 0, 1).squeeze(0)
        heatmap = torch.clamp( locnet(render), 0, 1).squeeze(0)
        heatmap = heatmap * mask

        locations = heatmap2loc(heatmap, threshold=THRESHOLD)
        if len(locations) > 0:
            windows = loc2window(render, locations)
            logits  = labelnet(windows)
            labels  = torch.argmax(logits, dim=-1)
            
            for location, label in zip(locations, labels):
                center = (location[1], location[0])
                color  = LABEL2COLOR[label.item()]

                cv2.circle(capture, center, radius=8, color=color, thickness=2)

        quad_v["texture"] = capture
        quad_m["texture"] =    mask.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy()
        quad_h["texture"] = heatmap.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy()
        
        window.clear()
        quad_v.draw(gl.GL_TRIANGLE_STRIP)
        quad_m.draw(gl.GL_TRIANGLE_STRIP)
        quad_h.draw(gl.GL_TRIANGLE_STRIP)


@window.event
def on_exit():
    camera.stop()


if __name__ == "__main__":
    app.use("pyglet")
    app.run(framerate=120)