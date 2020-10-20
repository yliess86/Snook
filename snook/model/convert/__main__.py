from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from snook.config import Config
from snook.model.convert.dir import DirNetConvertor
from snook.model.convert.label import LabelNetConvertor
from snook.model.convert.loc import LocNetConvertor
from snook.model.convert.mask import MaskNetConvertor


SNOOK_CONVERTOR = """
 ______     __   __     ______     ______     __  __    
/\  ___\   /\ "-.\ \   /\  __ \   /\  __ \   /\ \/ /    
\ \___  \  \ \ \-.  \  \ \ \/\ \  \ \ \/\ \  \ \  _"-.  
 \/\_____\  \ \_\ "\_\  \ \_____\  \ \_____\  \ \_\ \_\ 
  \/_____/   \/_/ \/_/   \/_____/   \/_____/   \/_/\/_/ 

© Snook.ai

The Snook Convertor:
* Convert Pretrained Snook Models '.pt' to TorchScript '.ts'                           
* Convert Pretrained Snook Models '.pt' to Onnx '.nx' and '.qnx'   
* Assert Equivalence of Backends                           
"""

parser = ArgumentParser(description=SNOOK_CONVERTOR, formatter_class=RawTextHelpFormatter)
parser.add_argument("-c", "--config", type=str, required=True, help="config yaml file")
parser.add_argument("-m", "--model",  type=str, required=True, help="directory to save the model conversion files")
parser.add_argument("-d", "--debug",  action="store_true",     help="show output image debug after each convertion")

args = parser.parse_args()

print(SNOOK_CONVERTOR)

config = Config.from_yaml(args.config)

convertor = LocNetConvertor(config.model.locnet, config.trainer.locnet, args.model)
convertor(debug=args.debug)

convertor = MaskNetConvertor(config.model.masknet, config.trainer.masknet, args.model)
convertor(debug=args.debug)

convertor = LabelNetConvertor(config.model.labelnet, config.trainer.labelnet, args.model)
convertor()

convertor = DirNetConvertor(config.model.dirnet, config.trainer.dirnet, args.model)
convertor()