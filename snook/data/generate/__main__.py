import io
import os

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from contextlib import redirect_stdout
from snook.config import Config
from snook.data.generate.generator import DataGenerator
from tqdm import tqdm


SNOOK_GENERATOR = """
 ______     __   __     ______     ______     __  __    
/\  ___\   /\ "-.\ \   /\  __ \   /\  __ \   /\ \/ /    
\ \___  \  \ \ \-.  \  \ \ \/\ \  \ \ \/\ \  \ \  _"-.  
 \/\_____\  \ \_\ "\_\  \ \_____\  \ \_____\  \ \_\ \_\ 
  \/_____/   \/_/ \/_/   \/_____/   \/_____/   \/_/\/_/ 

© Snook.ai

The Snook Dataset Generator produces:
  * Renders of Pool/Snooker Scenes (Image: PNG)
  * Balls, Cues, and Cue Targets Positions (Data: YAML)                               
"""

parser = ArgumentParser(description=SNOOK_GENERATOR, formatter_class=RawTextHelpFormatter)
parser.add_argument("-c", "--config",  type=str, required=True, help="config yaml file")
parser.add_argument("-r", "--render",  type=str, required=True, help="path to save the image files")
parser.add_argument("-d", "--data",    type=str, required=True, help="path to save the data files")
parser.add_argument("-s", "--samples", type=int, required=True, help="number of samples")

args = parser.parse_args()

print(SNOOK_GENERATOR)

print("[Snook][Generator] Creating Render and Data Directories")
os.makedirs(args.render, exist_ok=True)
os.makedirs(args.data, exist_ok=True)

print("[Snook][Generator] Initializing Generator")
with redirect_stdout(io.StringIO()):
  generator = DataGenerator(Config.from_yaml(args.config))

for i in tqdm(range(args.samples), desc="[Snook][Generator] Generating Render and Data"):
    max_n = len(str(args.samples))
    render = os.path.join(args.render, f"{i:0{max_n}d}.png")
    data = os.path.join(args.data, f"{i:0{max_n}d}.yaml")
    
    with redirect_stdout(io.StringIO()):
      generator(render, data)