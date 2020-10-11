import bpy
import bpy_extras as bpye
import numpy as np

from mathutils import Vector
from typing import Any
from typing import Dict
from typing import Optional


class Obj:
    def __init__(self, name: str) -> None:
        self.name = name
        self.obj = bpy.data.objects[self.name]

        self.position = np.zeros((3, ))
        self.rotation = np.zeros((3, ))

    @property
    def position(self) -> np.ndarray:
        return np.array(self.obj.location)

    @position.setter
    def position(self, value: np.array) -> None:
        self.obj.location = value

    @property
    def rotation(self) -> np.ndarray:
        return np.array(self.obj.rotation_euler)

    @rotation.setter
    def rotation(self, value: np.ndarray) -> None:
        self.obj.rotation_euler = value

    @property
    def visible(self) -> bool:
        return not self.obj.hide_render

    @visible.setter
    def visible(self, value: bool) -> None:
        self.obj.hide_render = not value
        self.obj.hide_viewport = not value

    def look_at(self, obj: "Obj", track: str, up: str) -> None:
        direction = obj.position - self.position
        direction /= np.linalg.norm(direction)
        quaternion = Vector(direction).to_track_quat(track, up)
        self.rotation = quaternion.to_euler()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
                f"name={self.name}, "
                f"pos={tuple(self.position)}, rot={tuple(self.rotation)}, "
                f"visible={self.visible}"
            f")"
        )


class FbxObj(Obj):
    def __init__(self, fbx_path: str, id: Optional[int] = 0) -> None:
        name = fbx_path.split("/")[-1].replace(".fbx", "")
        name_w_id = f"{name}_{id:03d}"
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        bpy.data.objects[name].name = name_w_id
        super(FbxObj, self).__init__(name_w_id)

    def replace_glass(self, name: str, roughness: float) -> None:
        mat = self.obj.material_slots[name].material
        mat.use_nodes = True
        mat.node_tree.nodes.new(type="ShaderNodeBsdfGlass")
        mat_surf = mat.node_tree.nodes["Material Output"].inputs["Surface"]
        mat_bsdf = mat.node_tree.nodes["Glass BSDF"].outputs["BSDF"]
        mat.node_tree.links.new(mat_surf, mat_bsdf)
        mat.node_tree.nodes["Glass BSDF"].inputs[1].default_value = roughness


class EmptyObj(Obj):
    def __init__(self, name: str, id: Optional[int] = 0) -> None:
        name_w_id = f"{name}_{id:03d}"
        bpy.ops.object.empty_add(type="PLAIN_AXES", align="WORLD")
        bpy.data.objects["Empty"].name = name_w_id
        super(EmptyObj, self).__init__(name_w_id)


class CameraObj(Obj):
    def __init__(self, name: str, id: Optional[int] = 0) -> None:
        name_w_id = f"{name}_{id:03d}"
        self.cam = bpy.data.cameras.new(name_w_id)
        bpy.data.objects.new(name_w_id, self.cam)
        super(CameraObj, self).__init__(name_w_id)
        self.scene = bpy.context.scene
        self.scene.camera = self.obj

    @property
    def sensor_fit(self) -> float:
        return self.cam.sensor_fit

    @sensor_fit.setter
    def sensor_fit(self, value: float) -> None:
        self.cam.sensor_fit = value

    @property
    def sensor_width(self) -> float:
        return self.cam.sensor_width

    @sensor_width.setter
    def sensor_width(self, value: float) -> None:
        self.cam.sensor_width = value

    @property
    def sensor_height(self) -> float:
        return self.cam.sensor_height

    @sensor_height.setter
    def sensor_height(self, value: float) -> None:
        self.cam.sensor_height = value

    @property
    def lens(self) -> float:
        return self.cam.lens

    @lens.setter
    def lens(self, value: float) -> None:
        self.cam.lens = value

    @property
    def dof(self) -> "Dof":
        return self.cam.dof

    def ndc(self, obj: Obj) -> np.ndarray:
        coord_2d = bpye.object_utils.world_to_camera_view(self.scene, self.obj, Vector(obj.position))
        render_scale = self.scene.render.resolution_percentage / 100
        render_w = int(self.scene.render.resolution_x * render_scale)
        render_h = int(self.scene.render.resolution_y * render_scale)
        return np.array([int(coord_2d.x * render_w), int((1 - coord_2d.y) * render_h)])
