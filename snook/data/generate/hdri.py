import bpy
import os
import numpy as np

from typing import Tuple


class EnvironmentMaps:
    def __init__(self) -> None:
        scene = bpy.context.scene
        scene.world.use_nodes = True
        node_tree = scene.world.node_tree

        texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        mapping_node = node_tree.nodes.new(type="ShaderNodeMapping")
        uvcoord_node = node_tree.nodes.new(type="ShaderNodeTexCoord")
        backgro_node = node_tree.nodes["Background"]

        node_tree.links.new(uvcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        node_tree.links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])
        node_tree.links.new(texture_node.outputs["Color"], backgro_node.inputs["Color"])

        self.hdri = texture_node
        self.map = mapping_node
        self.uv = uvcoord_node
        self.bkg = backgro_node

    @property
    def rotation(self) -> np.ndarray:
        return np.array(self.map.inputs["Rotation"].default_value)

    @rotation.setter
    def rotation(self, value: np.ndarray) -> None:
        self.map.inputs["Rotation"].default_value = value

    @property
    def scale(self) -> np.ndarray:
        return np.array(self.map.inputs["Scale"].default_value)

    @scale.setter
    def scale(self, value: np.ndarray) -> None:
        self.map.inputs["Scale"].default_value = value

    @property
    def image(self):
        return self.hdri.image

    @image.setter
    def image(self, value: str) -> None:
        self.hdri.image = bpy.data.images.load(value, check_existing=True)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(rot={self.rotation}, scale={self.scale})"