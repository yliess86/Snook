import cv2
import os
import torch

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from glumpy import app, gl, gloo
from snook.demo.camera import Camera
from snook.demo.colors import LABEL2COLOR
from snook.demo.inference import Snook


SNOOK_DEMO = """
 ______     __   __     ______     ______     __  __    
/\  ___\   /\ "-.\ \   /\  __ \   /\  __ \   /\ \/ /    
\ \___  \  \ \ \-.  \  \ \ \/\ \  \ \ \/\ \  \ \  _"-.  
 \/\_____\  \ \_\ "\_\  \ \_____\  \ \_____\  \ \_\ \_\ 
  \/_____/   \/_/ \/_/   \/_____/   \/_____/   \/_/\/_/ 

© Snook.ai

The Snook Demo
"""

VS = """
attribute vec2 pos; attribute vec2 uv; varying vec2 v_uv;
void main() { gl_Position = vec4(pos, 0., 1.); v_uv = uv; }
"""
FS = """
uniform sampler2D texture; varying vec2 v_uv;
void main() { gl_FragColor = texture2D(texture, v_uv).rgba; }
"""

LABEL_THRESH = 0.7
CAMERA_SIZE  = (1280, 720)
LOC_SIZE     = ( 256, 256)
MASK_SIZE    = (  64,  64)
APP_SIZE     = CAMERA_SIZE
QUAD_POS     = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)][::-1]
QUAD_UV      = [(+0, +0), (+0, +1), (+1, +0), (+1, +1)][::-1]


parser = ArgumentParser(description=SNOOK_DEMO, formatter_class=RawTextHelpFormatter)
parser.add_argument("-m", "--model",  type=str, required=True, help="directory to find model files")
parser.add_argument("-c", "--camera", type=int, required=True, help="camera device id")

args = parser.parse_args()

snook  = Snook(*(os.path.join(args.model, file) for file in ["locnet.ts", "masknet.ts", "labelnet.ts"])).eval().cuda()
camera = Camera(args.camera, *CAMERA_SIZE)
quad   = gloo.Program(VS, FS, count=4)
window = app.Window(width=APP_SIZE[0], height=APP_SIZE[1], title="Snook")


@window.event
def on_init():
    frac = 2 / 3
    quad["pos"] = QUAD_POS
    quad["uv"]  = QUAD_UV
    camera.start()


@window.event
def on_draw(dt):
    _, capture = camera.read()

    with torch.no_grad():
        x = torch.from_numpy(capture).float().permute((2, 0, 1)).unsqueeze(0).cuda()
        heatmap, coords, logits, labels = snook(x)
        if coords is not None:
            for coord, logit, label in zip(coords, logits, labels):
                if logit[label] > LABEL_THRESH:
                    x = (coord[1] * (CAMERA_SIZE[0] / LOC_SIZE[0])).int().item()
                    y = (coord[0] * (CAMERA_SIZE[1] / LOC_SIZE[1])).int().item()
                    cv2.circle(capture, (x, y), radius=25, color=LABEL2COLOR[label.item()], thickness=4)
        
        quad["texture"] = capture[::-1] 
        window.clear()
        quad.draw(gl.GL_TRIANGLE_STRIP)
        

@window.event
def on_exit():
    camera.stop()


print(SNOOK_DEMO)
app.use("pyglet")
app.run(framerate=120)