import cv2
import numpy as np
import os
import torch

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from glumpy import app, gl, gloo
from snook.demo.camera import Camera
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

HEATMAP_ALPHA = 0.6
MASK_ALPHA    = 0.8
LOGITS_ALPHA  = 0.8
ALPHAS        = HEATMAP_ALPHA, MASK_ALPHA, LOGITS_ALPHA

CAMERA_SIZE  = np.array((960, 540))
LOC_SIZE     = np.array((256, 256))
MASK_SIZE    = np.array(( 64,  64))
APP_SIZE     = CAMERA_SIZE
QUAD_POS     = np.array([(-1, -1), (-1, +1), (+1, -1), (+1, +1)][::-1])
QUAD_UV      = np.array([(+0, +0), (+0, +1), (+1, +0), (+1, +1)][::-1])


parser = ArgumentParser(description=SNOOK_DEMO, formatter_class=RawTextHelpFormatter)
parser.add_argument("-m", "--model",  type=str, required=True, help="directory to find model files")
parser.add_argument("-c", "--camera", type=int, required=True, help="camera device id")

args = parser.parse_args()

nets   = ["locnet.ts", "masknet.ts", "labelnet.ts", "dirnet.ts"]
snook  = Snook(*(os.path.join(args.model, file) for file in nets)).eval().cuda()
camera = Camera(args.camera, *CAMERA_SIZE)
quad   = gloo.Program(VS, FS, count=4)
window = app.Window(width=APP_SIZE[0], height=APP_SIZE[1], title="Snook")


@window.event
def on_init():
    quad["pos"] = QUAD_POS
    quad["uv"]  = QUAD_UV
    camera.start()


@window.event
def on_draw(dt):
    _, capture = camera.read()

    with torch.no_grad():
        x = torch.tensor(capture).float()
        x = x.permute((2, 0, 1)).unsqueeze(0).cuda()
        pool = snook(x, alphas=ALPHAS)
        
    pool.draw(capture, scale=CAMERA_SIZE / LOC_SIZE)
    quad["texture"] = capture[::-1]
    window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)
        

@window.event
def on_close():
    camera.stop()


print(SNOOK_DEMO)
app.use("pyglet")
app.run(framerate=120)