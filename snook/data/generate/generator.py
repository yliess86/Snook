import bpy
import numpy as np

from snook.config import Config
from snook.data.generate.data import PoolData
from snook.data.generate.obj import FbxObj
from snook.data.generate.obj import EmptyObj
from snook.data.generate.obj import CameraObj
from snook.data.generate.hdri import EnvironmentMaps
from snook.data.generate.render import CycleRenderer
from snook.data.generate.sampler import HalfOnionSkinSampler
from snook.data.generate.sampler import HDRISampler
from snook.data.generate.sampler import PoolSampler
from typing import NamedTuple


TOLERANCE = 1e-4
COLORS = ["black", "white", "red", "yellow"]


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2)) 


class Samplers(NamedTuple):
    pool: PoolSampler
    onion: HalfOnionSkinSampler
    hdris: HDRISampler


class DataGenerator:
    def __init__(self, conf: Config) -> None:
        self.resources_conf = conf.resources
        self.render_conf = conf.render
        self.camera_conf = conf.camera
        self.generator_conf = conf.generator

        self._excomuniate_default_cube()
        self._setup_pool()
        self._setup_cues()
        self._setup_balls()
        self._setup_hdri()
        self._setup_camera()
        self._setup_renderer()
        self._setup_samplers()
        self._setup_pool_data()

    def _excomuniate_default_cube(self) -> None:
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj)

    def _setup_pool(self) -> None:
        self.pool = FbxObj(self.resources_conf.pool)
        self.pool.replace_glass("Glass", roughness=0.15)
        self.pool_corners = [EmptyObj("pool_corner", i) for i in range(4)]

        w  = self.generator_conf.pool_half_width
        h  = self.generator_conf.pool_half_height
        z  = self.generator_conf.pool_depth
        bw = self.generator_conf.pool_border_width
        bh = self.generator_conf.pool_border_height
        self.pool_corners[0].position = np.array([-w - bw,  h + bh, z])
        self.pool_corners[1].position = np.array([ w + bw,  h + bh, z])
        self.pool_corners[2].position = np.array([ w + bw, -h - bh, z])
        self.pool_corners[3].position = np.array([-w - bw, -h - bh, z])
        
    def _setup_cues(self) -> None:
        self.cues = [FbxObj(self.resources_conf.cue, i) for i in range(6)]
        self.cues_target = [EmptyObj("cue_target", i) for i in range(6)]

    def _setup_balls(self) -> None:
        self.balls = []
        self.balls += [FbxObj(self.resources_conf.black_ball,  i) for i in range(6)]
        self.balls += [FbxObj(self.resources_conf.white_ball,  i) for i in range(6)]
        self.balls += [FbxObj(self.resources_conf.yellow_ball, i) for i in range(6)]
        self.balls += [FbxObj(self.resources_conf.red_ball,    i) for i in range(6)]

    def _setup_hdri(self) -> None:
        self.hdri = EnvironmentMaps()

    def _setup_camera(self) -> None:
        self.cam = CameraObj("cam")
        self.cam_target = EmptyObj("cam_target")
        self.cam.sensor_fit = self.camera_conf.sensor_fit
        self.cam.sensor_width = self.camera_conf.sensor_width
        self.cam.sensor_height = self.camera_conf.sensor_height
        self.cam.lens = self.camera_conf.lens
        self.cam.dof.use_dof = True
        self.cam.dof.focus_object = self.cam_target.obj
        self.cam.dof.aperture_fstop = self.camera_conf.dof_aperture_fstop
        self.cam.dof.aperture_blades = self.camera_conf.dof_aperture_blades

    def _setup_renderer(self) -> None:
        render_size = (self.render_conf.width, self.render_conf.height)
        tile_size = (self.render_conf.tile, self.render_conf.tile)
        self.renderer = CycleRenderer(render_size, tile_size, self.render_conf.samples)
        self.renderer.assign_camera(self.cam)
        self.renderer.denoise = self.render_conf.denoise

    def _setup_samplers(self) -> None:
        pool = self.generator_conf.pool_half_width, self.generator_conf.pool_half_height, self.generator_conf.pool_depth
        self.samplers = Samplers(
            PoolSampler(*pool, self.generator_conf.ball_radius),
            HalfOnionSkinSampler(np.zeros((3, )), (0, 0)),
            HDRISampler(self.resources_conf.hdris),
        )

    def _setup_pool_data(self) -> None:
        self.pool_data = PoolData()

    def _sample_balls(self) -> None:
        self.samplers.pool.reset()
        for ball in self.balls:
            ball.visible = np.random.rand() < self.generator_conf.ball_on_pool
            if ball.visible:
                ball.position = self.samplers.pool.sample()
                ball.position += np.array([0, 0, self.generator_conf.ball_radius])
                ball.rotation = np.random.uniform(0, 2 * np.pi, (3, ))

    def _sample_cues(self) -> None:
        for cue, cue_target in zip(self.cues, self.cues_target):
            cue.visible = np.random.rand() < self.generator_conf.cue_is_visible
            if cue.visible:
                balls = [ball for ball in self.balls if ball.visible]
                if np.random.rand() < self.generator_conf.cue_on_ball and len(balls) > 0:
                    ball = self.balls[np.random.choice(len(self.balls))]
                    radius = self.generator_conf.ball_radius
                    cue_target.position = ball.position
                    self.samplers.onion.radius_range = (radius, radius + self.generator_conf.cue_distance_max)
                else:
                    self.samplers.pool.reset()
                    cue_target.position = self.samplers.pool.sample()
                    radius = self.generator_conf.ball_radius
                    self.samplers.onion.radius_range = (radius, radius + self.generator_conf.cue_distance_max)

                self.samplers.onion.pos = cue_target.position + np.array([0, 0, self.generator_conf.ball_radius])
                ball_radius   = self.generator_conf.ball_radius
                out_borders_w = np.abs(self.samplers.onion.pos[0]) > self.generator_conf.pool_half_width  - ball_radius
                out_borders_h = np.abs(self.samplers.onion.pos[1]) > self.generator_conf.pool_half_height - ball_radius
                if out_borders_w and out_borders_h:
                    self.samplers.onion.pos += np.array([0, 0, ball_radius * 4])
                if self.samplers.onion.pos[2] <= self.generator_conf.pool_depth:
                    self.samplers.onion.pos[2] = self.generator_conf.pool_depth

                cue.position = self.samplers.onion.sample()

                in_pool_width = -self.generator_conf.pool_half_width < cue.position[0] < self.generator_conf.pool_half_width
                in_pool_height = -self.generator_conf.pool_half_height < cue.position[1] < self.generator_conf.pool_half_height
                if not in_pool_width or not in_pool_height:
                    cue.position[2] += self.generator_conf.ball_radius * 4
                
                cue.look_at(cue_target, "-Z", "Y")

    def _sample_cam(self) -> None:
        def sample_center(alpha: float) -> np.ndarray:
            self.samplers.pool.reset()
            position = self.samplers.pool.sample()
            position[:2] *= alpha
            return position

        alpha = 0.2
        
        self.samplers.onion.pos = sample_center(alpha)
        self.samplers.onion.pos[2] = np.random.uniform(*self.generator_conf.cam_dist_range)  
        self.samplers.onion.radius_range = (0, alpha)
        self.cam.position = self.samplers.onion.sample()

        self.samplers.onion.pos = sample_center(alpha) 
        self.samplers.onion.radius_range = (0, alpha)
        self.cam_target.position = self.samplers.onion.sample()

        self.cam.look_at(self.cam_target, "-Z", "Y")

    def _sample_hdri(self) -> None:
        scale = np.random.uniform(*self.generator_conf.hdri_scale_range)
        self.hdri.rotation = np.array([0, 0, np.random.uniform(0, 2 * np.pi)])
        self.hdri.scale = np.array([scale for i in range(3)])
        self.hdri.image = self.samplers.hdris.sample()

    def sample(self) -> None:
        self._sample_balls()
        self._sample_cues()
        self._sample_cam()
        self._sample_hdri()

    def render(self, path: str) -> None:
        self.renderer.render(path)

    def _register_balls(self) -> None:
        def ball_color(ball) -> str:
            for c in COLORS:
                if c in ball.name:
                    return c
            return None

        for ball in self.balls:
            if ball.visible:
                position = self.cam.ndc(ball)
                color = ball_color(ball)
                self.pool_data.append_ball(position.tolist(), color)

    def _register_cues(self) -> None:
        for cue, target in zip(self.cues, self.cues_target):
            if cue.visible:
                in_pool_w = np.abs(cue.position[0]) <= self.generator_conf.pool_half_width  + self.generator_conf.pool_border_width
                in_pool_h = np.abs(cue.position[1]) <= self.generator_conf.pool_half_height + self.generator_conf.pool_border_height
                if in_pool_w and in_pool_h:
                    position  = self.cam.ndc(cue)
                    in_width  = (position[0] >= 0) and (position[0] < self.render_conf.width)
                    in_height = (position[1] >= 0) and (position[1] < self.render_conf.height)
                    if in_width and in_height:
                        target_position = self.cam.ndc(target).tolist()
                        self.pool_data.append_cue(position.tolist(), target_position)

    def _register_pool(self) -> None:
        self.pool_data.append_table(*[self.cam.ndc(corner).tolist() for corner in self.pool_corners])
        
    def register(self, path: str) -> None:
        self.pool_data.reset()
        self._register_balls()
        self._register_cues()
        self._register_pool()
        self.pool_data.save(path)

    def __call__(self, render_path: str, pool_data_path: str) -> None:
        self.sample()
        self.render(render_path)
        self.register(pool_data_path)
