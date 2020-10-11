import bpy

from snook.data.generate.obj import CameraObj
from typing import Tuple


Size = Tuple[int, int]


class CycleRenderer:
    def __init__(self, size: Size, tile_size: Size, samples: int) -> None:
        self.size = size
        self.tile_size = tile_size
        self.samples = samples
        self.scene = bpy.context.scene
        self.prefs = bpy.context.preferences.addons["cycles"].preferences
        self._setup()

    def assign_camera(self, cam: CameraObj) -> None:
        self.scene.camera = cam.obj

    @property
    def denoise(self) -> bool:
        return self.scene.cycles.use_denoising

    @denoise.setter
    def denoise(self, value: bool) -> None:
         self.scene.cycles.use_denoising = value

    def _setup(self) -> None:
        cuda_devices, _ = self.prefs.get_devices()
        for device in cuda_devices:
            device.use = True
        self.prefs.compute_device_type = "CUDA"
        bpy.ops.wm.save_userpref()

        self.scene.render.image_settings.file_format = "PNG"
        self.scene.render.engine = "CYCLES"
        self.scene.render.use_motion_blur = False
        self.scene.render.film_transparent = False
        self.scene.render.resolution_x = self.size[0]
        self.scene.render.resolution_y = self.size[1]
        self.scene.render.tile_x = self.tile_size[0]
        self.scene.render.tile_y = self.tile_size[1]

        self.scene.cycles.samples = self.samples
        self.scene.cycles.device = "GPU"
        self.scene.cycles.feature_set = "EXPERIMENTAL"
        self.scene.cycles.use_adaptive_sampling = True
        self.scene.cycles.use_denoising = True
        self.scene.cycles.denoiser = "NLM"

        self.scene.view_settings.view_transform = "Filmic"
        self.scene.view_settings.look = "High Contrast"

    def render(self, path: str) -> None:
        self.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
                f"size={self.size}, tile={self.tile_size}, "
                f"samples={self.samples}, "
                f"camera={self.scene.camera is not None}"
            f")"
        )