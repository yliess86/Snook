import numpy as np
import onnx as nx
import onnxruntime as nxruntime
import os
import torch
import torch.onnx as tonnx

from onnxruntime.quantization import quantize_dynamic
from onnx.utils import polish_model
from PIL import Image
from snook.config import AttributeDict
from snook.data.dataset.label import LabelDataset
from snook.model.convert.base import Convertor
from snook.model.network.label import LabelNet
from snook.model.network.modules import ConvBN2d
from tabulate import tabulate
from typing import Dict


class LabelNetConvertor(Convertor):
    def __init__(self, model: AttributeDict, data: AttributeDict, root: str) -> None:
        self.outputs: Dict[str, np.ndarray] = {}
        self.root = root
        self.model_conf = model

        print("[Loc][Convertor] Preparing Inputs")
        render, *_ = LabelDataset(data.dataset.test_render, data.dataset.test_data, train=False)[0]
        self.torch_render = render.unsqueeze(0)
        self.numpy_render = self.torch_render.detach().cpu().numpy()

    def vanilla(self) -> None:
        print("[Label][Convertor][Vanilla] Loading Pretrained Model")
        ckpt = os.path.join(self.root, "snook.pt")
        self.model = LabelNet.from_config(self.model_conf)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu')["labelnet"])
        self.model.cpu().eval()
        self.model.fuse()

        print("[Label][Convertor][Vanilla] Computing Output")
        with torch.no_grad():
            self.outputs["Vanilla"] = self.model(self.torch_render).cpu().numpy()

    def jit(self) -> None:
        print("[Label][Convertor][TorchScript] Converting Model")
        ckpt = os.path.join(self.root, "labelnet.ts")
        torch.jit.save(torch.jit.trace(self.model, self.torch_render), ckpt)
        
        model = torch.jit.load(ckpt)
        with torch.no_grad():
            self.outputs["TorchScript"] = model(self.torch_render).cpu().numpy()
   
    def onnx(self, ops: int = 11) -> None:        
        print("[Label][Convertor][Onnx] Converting Model")
        ckpt = os.path.join(self.root, "labelnet.nx")
        params   = [name for name, _ in self.model.named_parameters()]
        dynamics = { "img" : { 0 : "batch_size" }, "heatmaps" : { 0 : "batch_size" } }
        in_outs  = { "input_names": ["img"] + params, "output_names": ["heatmaps"], "dynamic_axes": dynamics }
        options  = { "export_params": True, "keep_initializers_as_inputs": True }
        tonnx.export(self.model, self.torch_render, ckpt, opset_version=ops, **in_outs, **options)

        model = nx.load(ckpt)
        model = polish_model(model)

        inputs = model.graph.input
        name2inputs = {} 
        for input in inputs:
            name2inputs[input.name] = input
        for initializer in model.graph.initializer:
            if initializer.name in name2inputs:
                inputs.remove(name2inputs[initializer.name])
        nx.save(model, ckpt)

        qckpt = os.path.join(self.root, "labelnet.qnx")
        quantize_dynamic(ckpt, qckpt, per_channel=True)

        sess = nxruntime.InferenceSession(ckpt)
        sess.set_providers(["CPUExecutionProvider"])
        inputs = sess.get_inputs()[0].name
        self.outputs["Onnx"] = sess.run(None, { inputs: self.numpy_render })[0]

        sess = nxruntime.InferenceSession(qckpt)
        sess.set_providers(["CPUExecutionProvider"])
        inputs = sess.get_inputs()[0].name
        self.outputs["Quantized Onnx"] = sess.run(None, { inputs: self.numpy_render })[0]

    def compare(self) -> None:
        diff = lambda a, b: np.max(np.abs(a - b))
        backends = list(self.outputs.keys())
        data = ["Vanilla"] + [diff(self.outputs["Vanilla"], self.outputs[b]) for b in backends]
        table = tabulate([data], headers=["Backend"] + backends)
        print("[Label][Convertor] Max Absolute Backends Differences from Vanilla PyTorch\n")
        print(table, "\n")

    def debug(self) -> None:
        pass