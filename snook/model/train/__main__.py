import numpy as np
import os
import torch

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from snook.config import Config
from snook.model.train.dir import DirSyntheticTrainer
from snook.model.train.label import LabelSyntheticTrainer
from snook.model.train.loc import LocSyntheticTrainer
from snook.model.train.mask import MaskSyntheticTrainer


SNOOK_TRAINER = """
 ______     __   __     ______     ______     __  __    
/\  ___\   /\ "-.\ \   /\  __ \   /\  __ \   /\ \/ /    
\ \___  \  \ \ \-.  \  \ \ \/\ \  \ \ \/\ \  \ \  _"-.  
 \/\_____\  \ \_\ "\_\  \ \_____\  \ \_____\  \ \_\ \_\ 
  \/_____/   \/_/ \/_/   \/_____/   \/_____/   \/_/\/_/ 

© Snook.ai

The Snook Trainer:
* Train Snook Models with Synthetic Images from the Snook Dataset
* Saves Snook Models as '.pt' File
* Snook Models:
    * LocNet   (Heatmap Ball & Cue Localization)
    * MaskNet  (Pool Table Mask)
    * LabelNet (Ball Color Classifier)
    * DirNet   (Cue Direction Regressor)
"""


parser = ArgumentParser(description=SNOOK_TRAINER, formatter_class=RawTextHelpFormatter)
parser.add_argument("-c", "--config", type=str, required=True, help="config yaml file")
parser.add_argument("-m", "--model",  type=str, required=True, help="directory to save the model files")
parser.add_argument("-d", "--debug",  action="store_true",     help="show first 3 test images locnet outputs after training")

args = parser.parse_args()

print(SNOOK_TRAINER)

conf = Config.from_yaml(args.config)

trainer = LocSyntheticTrainer(conf.model.locnet, conf.trainer.locnet)
trainer(epochs=conf.trainer.locnet.hyperparameters.epochs, debug=args.debug)
locnet = trainer.model
print(f"[LocNet][Trainer][Synthetic] Saving Model")
torch.save(locnet.eval().state_dict(), os.path.join(args.model, "locnet.pt"))

trainer = MaskSyntheticTrainer(conf.model.masknet, conf.trainer.masknet)
trainer(epochs=conf.trainer.masknet.hyperparameters.epochs, debug=args.debug)
masknet = trainer.model
print(f"[MaskNet][Trainer][Synthetic] Saving Model")
torch.save(masknet.eval().state_dict(), os.path.join(args.model, "masknet.pt"))

trainer = LabelSyntheticTrainer(conf.model.labelnet, conf.trainer.labelnet)
trainer(epochs=conf.trainer.labelnet.hyperparameters.epochs)
labelnet = trainer.model
print(f"[LabelNet][Trainer][Synthetic] Saving Model")
torch.save(labelnet.eval().state_dict(), os.path.join(args.model, "labelnet.pt"))

trainer = DirSyntheticTrainer(conf.model.dirnet, conf.trainer.dirnet)
trainer(epochs=conf.trainer.dirnet.hyperparameters.epochs)
dirnet = trainer.model
print(f"[DirNet][Trainer][Synthetic] Saving Model")
torch.save(dirnet.eval().state_dict(), os.path.join(args.model, "dirnet.pt"))