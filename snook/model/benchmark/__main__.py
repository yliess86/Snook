from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from snook.model.benchmark.dir import DirBenchmark
from snook.model.benchmark.label import LabelBenchmark
from snook.model.benchmark.loc import LocBenchmark
from snook.model.benchmark.mask import MaskBenchmark


SNOOK_BENCHMARK = """
 ______     __   __     ______     ______     __  __    
/\  ___\   /\ "-.\ \   /\  __ \   /\  __ \   /\ \/ /    
\ \___  \  \ \ \-.  \  \ \ \/\ \  \ \ \/\ \  \ \  _"-.  
 \/\_____\  \ \_\ "\_\  \ \_____\  \ \_____\  \ \_\ \_\ 
  \/_____/   \/_/ \/_/   \/_____/   \/_____/   \/_/\/_/ 

© Snook.ai

The Snook Benchmark:
* Benchmark Snook Models FPS                          
* Runs Benchmark Multiple Time, Removes Outliers and Average                          
"""


parser = ArgumentParser(description=SNOOK_BENCHMARK, formatter_class=RawTextHelpFormatter)
parser.add_argument("-c", "--config",  type=str, required=True, help="config yaml file")
parser.add_argument("-m", "--model",   type=str, required=True, help="directory to find the model files")
parser.add_argument("-s", "--samples", type=int, required=True, help="benchmark samples")

args = parser.parse_args()

print(SNOOK_BENCHMARK)

benchmark = LocBenchmark(args.config, args.model)
benchmark(args.samples)

benchmark = MaskBenchmark(args.config, args.model)
benchmark(args.samples)

benchmark = LabelBenchmark(args.config, args.model)
benchmark(args.samples)

benchmark = DirBenchmark(args.config, args.model)
benchmark(args.samples)