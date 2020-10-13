import cpuinfo
import numpy as np
import onnxruntime as nx
import os
import torch

from snook.config import AttributeDict
from snook.config import Config
from snook.model.benchmark.base import Benchmark
from snook.model.network.mask import MaskNet
from tabulate import tabulate
from typing import Dict


class MaskBenchmark:
    def __init__(self, config: AttributeDict, root: str) -> None:
        self.config = config
        self.root = root
        self.benchmark = {}

    def __call__(self, samples: int = 200) -> None:
        self.vanilla(samples)
        self.jit(samples)
        self.onnx(samples)
        self.show(samples)

    def vanilla(self, samples: int) -> None:
        self.benchmark["Vanilla"] = {}
        ckpt = os.path.join(self.root, "snook.pt")
        conf = Config.from_yaml(self.config).model.locnet

        model = MaskNet.from_config(conf)
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["masknet"])

        model = model.cpu().eval()
        x = torch.randn((1, 3, 64, 64), requires_grad=False)
        with Benchmark("[MaskNet][Vanilla][CPU]") as bm:
            bm(samples, lambda: model(x))
            self.benchmark["Vanilla"]["CPU"] = bm.fps

        model = model.cuda().eval()
        x = x.cuda()
        with Benchmark("[MaskNet][Vanilla][CUDA]") as bm:
            bm(samples, lambda: model(x))
            self.benchmark["Vanilla"]["CUDA"] = bm.fps

    def jit(self, samples: int = 200) -> None:
        self.benchmark["TorchScript"] = {}
        ckpt = os.path.join(self.root, "masknet.ts")
        model = torch.jit.load(ckpt)

        model = model.cpu().eval()
        x = torch.randn((1, 3, 64, 64), requires_grad=False)
        with Benchmark("[MaskNet][TorchScript][CPU]") as bm:
            bm(samples, lambda: model(x))
            self.benchmark["TorchScript"]["CPU"] = bm.fps

        model = model.cuda().eval()
        x = x.cuda()
        with Benchmark("[MaskNet][TorchScript][CUDA]") as bm:
            bm(samples, lambda: model(x))
            self.benchmark["TorchScript"]["CUDA"] = bm.fps

    def onnx(self, samples: int = 200) -> None:
        self.benchmark["Onnx"] = {}
        x = np.random.random((1, 3, 64, 64)).astype(np.float32)
        
        ckpt = os.path.join(self.root, "masknet.nx")  
        sess = nx.InferenceSession(ckpt)
        providers = sess.get_providers()

        sess.set_providers(["CPUExecutionProvider"])
        inputs = sess.get_inputs()[0].name
        with Benchmark("[MaskNet][Onnx][CPU]") as bm:
            bm(samples, lambda: sess.run(None, { inputs: x }))
            self.benchmark["Onnx"]["CPU"] = bm.fps

        if "CUDAExecutionProvider" in providers:
            sess.set_providers(["CUDAExecutionProvider"])
            inputs = sess.get_inputs()[0].name
            with Benchmark("[MaskNet][Onnx][CUDA]") as bm:
                bm(samples, lambda: sess.run(None, { inputs: x }))
                self.benchmark["Onnx"]["CUDA"] = bm.fps

        self.benchmark["Quantized Onnx"] = {}

        ckpt = os.path.join(self.root, "masknet.qnx")  
        sess = nx.InferenceSession(ckpt)
        providers = sess.get_providers()

        sess.set_providers(["CPUExecutionProvider"])
        inputs = sess.get_inputs()[0].name
        with Benchmark("[MaskNet][Quantized Onnx][CPU]") as bm:
            bm(samples, lambda: sess.run(None, { inputs: x }))
            self.benchmark["Quantized Onnx"]["CPU"] = bm.fps

        if "CUDAExecutionProvider" in providers:
            sess.set_providers(["CUDAExecutionProvider"])
            inputs = sess.get_inputs()[0].name
            with Benchmark("[MaskNet][Quantized Onnx][CUDA]") as bm:
                bm(samples, lambda: sess.run(None, { inputs: x }))
                self.benchmark["Quantized Onnx"]["CUDA"] = bm.fps

    def show(self, samples: int) -> None:
        data = [(name, *list(bench.values())) for name, bench in self.benchmark.items()]
        table = tabulate(data, headers=["Backend", "CPU (FPS)", "CUDA (FPS)"])
        cuda = torch.cuda.get_device_name(torch.cuda.current_device())
        cpu = cpuinfo.get_cpu_info()["brand_raw"]
        print(f"[MaskNet][Benchmark] Results ({samples} samples)\n")
        print(table, "\n")
        print(f"{cpu} - {cuda}\n")