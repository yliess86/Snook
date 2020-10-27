from abc import ABC, abstractmethod
from mathutils import Euler, Vector
from typing import Optional, Tuple

import bpy
import bpy_extras as bpye


Size = Tuple[int, int]


def excomuniate_default_cube() -> None:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)


class Object:
    def __init__(self, name: str) -> None:
        self.name = name
        self.obj = bpy.data.objects[name]

    @property
    def pos(self) -> Vector:
        return self.obj.location

    @pos.setter
    def pos(self, pos: Vector) -> None:
        self.obj.location = pos

    @property
    def rot(self) -> Euler:
        return self.obj.rotation_euler

    @rot.setter
    def rot(self, rot: Euler) -> None:
        self.obj.rotation_euler = rot

    @property
    def visible(self) -> bool:
        return not self.obj.hide_render

    @visible.setter
    def visible(self, visible: bool) -> None:
        self.obj.hide_render = not visible
        self.obj.hide_viewport = not visible

    def look_at(self, target: Vector, track: str, up: str) -> None:
        direction = (target - self.pos).normalized()
        self.rot = direction.to_track_quat(track, up).to_euler()

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        infos = (
            f"name={self.name}",
            f"pos={self.pos}, rot={self.rot}, visible={self.visible}"
        )
        return f"{clsname}({infos})"


class Fbx(Object):
    def __init__(self, fbx: str, *, id: Optional[int] = 0) -> None:
        bpy.ops.import_scene.fbx(filepath=fbx)
        obj = bpy.context.selected_objects[0]
        obj.name = f"{obj.name}_{id}"
        super(Fbx, self).__init__(obj.name)


class Camera(Object):
    def __init__(
        self,
        name: str,
        *,
        sensor_fit: str,
        sensor_size: Tuple[float, float],
        lens: float,
        id: Optional[int] = 0,
    ) -> None:
        cam = bpy.data.cameras.new(f"{name}_{id}")
        obj = bpy.data.objects.new(f"{name}_{id}", cam)
        super(Camera, self).__init__(obj.name)
        self.cam = cam
        self.cam.sensor_fit = sensor_fit
        self.cam.sensor_width, self.cam.sensor_height = sensor_size
        self.cam.lens = lens

    def ndc(self, pos: Vector) -> Vector:
        scene = bpy.context.scene
        coords = bpye.object_utils.world_to_camera_view(scene, self.obj, pos)

        scale = scene.render.resolution_percentage / 100
        w = scene.render.resolution_x * scale
        h = scene.render.resolution_y * scale

        return Vector((coords.x * w, (1.0 - coords.y) * h, 0.0))
        

class HDRI:
    def __init__(self, img: str) -> None:
        scene = bpy.context.scene
        scene.world.use_nodes = True
        tree = scene.world.node_tree

        self.texture = tree.nodes.new(type="ShaderNodeTexEnvironment")
        self.mapping = tree.nodes.new(type="ShaderNodeMapping")
        self.uvs = tree.nodes.new(type="ShaderNodeTexCoord")
        self.bckg = tree.nodes["Background"]

        for inp, out in [
            (self.uvs.outputs["Generated"], self.mapping.inputs["Vector"]),
            (self.mapping.outputs["Vector"], self.texture.inputs["Vector"]),
            (self.texture.outputs["Color"], self.bckg.inputs["Color"]),
        ]:
            tree.links.new(inp, out)

        self.img = img

    @property
    def rot(self) -> Euler:
        return self.mapping.inputs["Rotation"].default_value

    @rot.setter
    def rot(self, rot: Euler) -> None:
        self.mapping.inputs["Rotation"].default_value = rot

    @property
    def scale(self) -> Vector:
        return self.mapping.inputs["Scale"].default_value

    @scale.setter
    def scale(self, scale: Vector) -> None:
        self.mapping.inputs["Scale"].default_value = scale

    @property
    def img(self) -> str:
        return self.texture.image.name

    @img.setter
    def img(self, img: str) -> None:
        self.texture.image = bpy.data.images.load(img, check_existing=True)

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        infos = f"img={self.img}, rot={self.rot}, scale={self.scale}"
        return f"{clsname}({infos})"


class Renderer(ABC):
    @abstractmethod
    def render(self, path: str) -> None:
        raise NotImplemented


class Cycle(Renderer):
    def __init__(
        self,
        resolution: Size,
        camera: Camera,
        *,
        tile: Size,
        samples: int,
        denoise: bool,
        cuda: bool,
    ) -> None:
        self.scene = bpy.context.scene
        self.prefs = bpy.context.preferences.addons["cycles"].preferences

        self.resolution = resolution
        self.tile = tile
        self.samples = samples
        self.denoise = denoise
        self.cuda = cuda
        
        self.scene.render.image_settings.file_format = "PNG"
        self.scene.render.engine = "CYCLES"
        self.scene.render.use_motion_blur = False
        self.scene.render.film_transparent = False
        self.scene.cycles.samples = self.samples
        self.scene.cycles.feature_set = "EXPERIMENTAL"
        self.scene.cycles.use_adaptive_sampling = True
        self.scene.view_settings.view_transform = "Filmic"
        self.scene.view_settings.look = "High Contrast"

        self.scene.camera = camera.obj

    @property
    def resolution(self) -> Size:
        return self.scene.render.resolution_x, self.scene.render.resolution_y

    @resolution.setter
    def resolution(self, resolution: Size) -> None:
        self.scene.render.resolution_x = resolution[0] 
        self.scene.render.resolution_y = resolution[1]

    @property
    def tile(self) -> Size:
        return self.scene.render.tile_x, self.scene.render.tile_y

    @tile.setter
    def tile(self, tile: Size) -> None:
        self.scene.render.tile_x, self.scene.render.tile_y = tile

    @property
    def denoise(self) -> bool:
        return self.scene.cycles.use_denoising

    @denoise.setter
    def denoise(self, denoise: bool) -> None:
        self.scene.cycles.use_denoising = denoise
        if denoise: self.scene.cycles.denoiser = "NLM"

    @property
    def cuda(self) -> bool:
        return self.scene.cycles.device == "GPU"

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        if cuda:
            for device in self.prefs.get_devices()[0]:
                device.use = True
            self.prefs.compute_device_type = "CUDA"
            bpy.ops.wm.save_userpref()
        self.scene.cycles.device = "GPU" if cuda else "CPU"

    def render(self, path: str) -> None:
        self.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        infos = (
            f"res={self.resolution}, "
            f"tile={self.tile}, samples={self.samples}, "
            f"cuda={self.cuda}, denoise={self.denoise}"
        )
        return f"{clsname}({infos})"