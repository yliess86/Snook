from mathutils import Euler, Vector
from torch.utils.data import DataLoader

import bpy
import math
import numpy as np
import os
import snook.data.blender as blender
import snook.data.generator as generator
import snook.data.dataset as dataset


def assert_vector_eq(a: Vector, b: Vector) -> bool:
    _a, _b = np.array(a), np.array(b)
    return np.testing.assert_almost_equal(_a, _b)


def assert_vector_range(v: Vector, a: float, b: float) -> bool:
    _v = np.array(v)
    return np.all((_v >= a) & (_v < b))


def assert_eurler_eq(a: Euler, b: Euler) -> bool:
    _a, _b = np.array(a), np.array(b)
    return np.testing.assert_almost_equal(_a, _b)


def assert_euler_range(e: Euler, a: float, b: float) -> bool:
    _e = np.array(e)
    return np.all((_e >= a) & (_e < b))


class TestDataBlender:
    def test_object(self) -> None:
        obj = blender.Object("Cube")
        assert obj.name == "Cube"

        assert_vector_eq(obj.pos, Vector())
        assert_eurler_eq(obj.rot, Euler())
        
        obj.visible = not obj.visible
        assert not obj.visible
        obj.visible = not obj.visible
        assert obj.visible

        cam = blender.Object("Camera")
        assert cam.name == "Camera"

        cam.pos = Vector((0.0, 1.0, 0.0))
        cam.rot = Euler()
        cam.look_at(obj.pos, "-Z", "Y")

        expected = (math.radians(90), 0, math.radians(180))
        assert_eurler_eq(cam.rot, Euler(expected))

    def test_excomuniate_default_cube(self) -> None:
        blender.excomuniate_default_cube()
        
        keys = list(bpy.data.objects.keys())
        for name in ["Cube", "Camera", "Light"]:
            assert "Cube" not in keys

        assert len(bpy.data.objects) <= 0

    def test_fbx(self) -> None:
        path = "./resources/fbx/ball_white.fbx"
        fbx_0 = blender.Fbx(path)
        fbx_1 = blender.Fbx(path, id=1)

        assert fbx_0.name == "ball_white_0"
        assert fbx_1.name == "ball_white_1"

    def test_camera(self) -> None:
        cam = blender.Camera(
            "Camera", sensor_fit="HORIZONTAL", sensor_size=(1, 1), lens=1.0
        )
        assert cam.name == "Camera_0"

        obj = blender.Object("ball_white_0")
        cam.pos = Vector((0.0, 1.0, 0.0))
        cam.look_at(obj.pos, "-Z", "Y")

        expected = (math.radians(90), 0, math.radians(180))
        assert_vector_eq(cam.pos, Vector((0.0, 1.0, 0.0)))
        assert_eurler_eq(cam.rot, expected)

        coords = cam.ndc(obj.pos)
        assert_vector_eq(coords, Vector((960.0, 540.0, 0.0)))

        occlusion_obj = blender.Object("ball_white_1")
        obj.pos = Vector((0, 0, 0))
        cam.pos = Vector((0, 2, 0))
        cam.look_at(obj.pos, "-Z", "Y")
        
        for pos in [(0, 0.5, 0), (0, 1.0, 0), (0, 1.5, 0)]:
            occlusion_obj.pos = Vector(pos)
            assert obj.occluded(cam, offset=0.1)

        for pos in [(0, -0.5, 0), (0.5, 0, 0), (0, 0, 0.5)]:
            occlusion_obj.pos = Vector(pos)
            assert not obj.occluded(cam, offset=0.1)
        
    def test_hdri(self) -> None:
        root = "./resources/hdri"
        files = os.listdir(root)

        name = files[0]
        hdri = blender.HDRI(os.path.join(root, name))
        assert hdri.img == name

        name = files[1]
        hdri.rot = Euler()
        hdri.scale = Vector((1.0, 1.0, 1.0))
        hdri.img = os.path.join(root, name)
        assert_eurler_eq(hdri.rot, Euler())
        assert_vector_eq(hdri.scale, Vector((1.0, 1.0, 1.0)))
        assert hdri.img == name

    def test_cycle(self, tmpdir) -> None:
        cam = blender.Object("Camera_0")
        cycle = blender.Cycle(
            (64, 64), cam, tile=(32, 32), samples=16, denoise=True, cuda=False
        )

        scene = bpy.context.scene
        assert scene.render.engine == "CYCLES"
        assert scene.cycles.device == "CPU"
        assert scene.cycles.use_denoising
        assert scene.render.resolution_x == 64
        assert scene.render.resolution_y == 64
        assert scene.render.tile_x == 32
        assert scene.render.tile_y == 32
        assert scene.cycles.samples == 16

        cycle.cuda = True
        cycle.denoise = False
        assert scene.cycles.device == "GPU"
        assert not scene.cycles.use_denoising

        cycle.render(str(tmpdir.mkdir("tmp").join("test_cycle.png")))


class TestDataGenerator:
    def test_init(self) -> None:
        blender.excomuniate_default_cube()

    def test_hdris(self) -> None:
        hdris = generator.Hdris(
            "./resources/hdri", rot=(0, 360), scale=(0.2, 2.0)
        )
        assert len(hdris) == 411
        assert os.path.isfile(hdris[0])

        for i in range(10):
            hdris.sample()
            hdri = hdris.hdri
            assert_vector_range(hdri.scale, 0.2, 2.0)
            assert_euler_range(hdri.rot, math.radians(0), math.radians(360))

    def test_pool(self) -> None:
        plane = generator.Plane((1.0, 2.0), 1.0)
        n = 2

        pos = plane.sample()
        assert abs(pos.x) <= 1.0 and abs(pos.y) <= 2.0 and pos.z == 1.0

        colors = generator.COLORS
        balls = generator.Balls(
            [f"./resources/fbx/ball_{color}.fbx" for color in colors],
            plane=plane,
            diameter=0.01,
            p=0.5,
            n=n,
        )
        
        assert len(balls) == n * 4
        for b, ball in enumerate(balls):
            idx = b % n
            color = colors[b // n]
            assert ball.name == f"ball_{color}_{idx}"
        for i in range(10): balls.sample()

        cues = generator.Cues(
            "./resources/fbx/cue.fbx",
            balls=balls,
            plane=plane,
            distance=2.0,
            p=0.5,
            n=n,
        )
        
        assert len(cues) == n
        for c, cue in enumerate(cues):
            assert cue.name == f"cue_{c}"
        for i in range(10): cues.sample()

        pool = generator.Pool(
            "./resources/fbx/pool.fbx", balls=balls, cues=cues
        )
        for i in range(10): pool.sample()

        cam = generator.Camera(
            blender.Camera(
                "Camera", sensor_fit="HORIZONTAL", sensor_size=(1, 1), lens=1.0
            ),
            plane=plane,
            distance=(5.0, 15.0),
            eps=1e-2,
        )
        for i in range(10): cam.sample()

        composition = generator.Samplers([pool, cam])
        for i in range(10): composition.sample()

    def test_scene(self, tmpdir) -> None:
        blender.excomuniate_default_cube()

        colors = generator.COLORS
        balls = [f"./resources/fbx/ball_{color}.fbx" for color in colors]
        cue = "./resources/fbx/cue.fbx"
        pool = "./resources/fbx/pool.fbx"
        hdri = "./resources/hdri"

        scene = generator.Scene(
            generator.cFiles(balls, cue, pool, hdri),
            generator.cTable((2.07793, 1.03677), (0.25, 0.20), 1.70342),
            generator.cDistances(0.1, 1.5, (10.0, 20.0)),
        )

        directory = tmpdir.mkdir("tmp")
        scene.sample()
        scene.render(str(directory.join("test_scene.png")))
        scene.register(str(directory.join("test_scene.txt")))


class TestDataDataset:
    CONTENT = """[24 balls] ndc_x ndc_y label
266 247 0
270 182 0
233 266 0
242 105 0
232 106 0
203 123 0
196 148 1
299 192 1
302 247 1
299 424 1
358 317 1
274 148 1
228 161 2
260 127 2
242 262 2
199 135 2
201 195 2
237 299 2
291 415 3
402 346 3
259 377 3
213 165 3
282 186 3
256 256 3

[6 cues] ndc_x ndc_y dir_x dir_y
110 138 0.9902610182762146 0.13922305405139923
334 357 -0.43943628668785095 0.8982738256454468
273 112 0.1970456838607788 0.9803943037986755
184 221 0.9992101192474365 -0.03973798826336861
208 180 0.9951098561286926 0.09877397865056992
256 256 -0.910141110420227 0.4142983555793762

[4 mask] ndc_x ndc_y
427 360
246 56
85 152
266 456"""

    def test_data_file_parser(self) -> None:
        data = balls, cues, mask = dataset.parse_data_file(self.CONTENT)
        assert len(balls) == 24
        assert len(cues) == 6
        assert len(mask) == 4

    def test_create_mask(self) -> None:
        mask = dataset.create_mask(
            (8, 8), corners=[(0, 0), (2, 0), (2, 2), (0, 2)]
        )
        assert mask.shape == (8, 8)
        assert np.all(mask[:3, :3] == 1.0) and np.all(mask[3:, 3:] == 0)

    def test_create_gaussian(self) -> None:
        gauss = dataset.create_gaussian((8, 8), point=(4, 4), spread=1.0)
        gauss /= gauss.max()
        assert gauss.shape == (8, 8)
        assert gauss[4, 4] == 1.0
        assert (gauss == 1.0).sum() == 1

    def test_create_heatmap(self) -> None:
        heatmap = dataset.create_heatmap(
            (16, 16), points=((4, 4), (8, 8)), spread=1.0
        )
        assert heatmap.shape == (16, 16)
        assert (heatmap[4, 4] == 1.0) and (heatmap[8, 8] == 1.0)
        assert (heatmap == 1.0).sum() == 2

    def test_datasets(self, tmpdir) -> None:
        blender.excomuniate_default_cube()

        colors = generator.COLORS
        balls = [f"./resources/fbx/ball_{color}.fbx" for color in colors]
        cue = "./resources/fbx/cue.fbx"
        pool = "./resources/fbx/pool.fbx"
        hdri = "./resources/hdri"

        scene = generator.Scene(
            generator.cFiles(balls, cue, pool, hdri),
            generator.cTable((2.07793, 1.03677), (0.25, 0.20), 1.70342),
            generator.cDistances(0.1, 1.5, (10.0, 20.0)),
        )

        directory = tmpdir.mkdir("tmp")
        renders = directory.mkdir("renders")
        data = directory.mkdir("data")
        for i in range(4):
            scene.sample()
            scene.render(str(renders.join(f"{i}.png")))
            scene.register(str(data.join(f"{i}.txt")))

        test_set = dataset.ReMaHeDataset(renders, data, spread=4.0)
        assert len(test_set) == 4

        loader = DataLoader(test_set, batch_size=4, shuffle=False)
        for (render, mask, heatmap) in loader:
            assert tuple(render.size()) == (4, 3, 512, 512)
            assert tuple(mask.size()) == (4, 512, 512)
            assert tuple(heatmap.size()) == (4, 512, 512)

        test_set = dataset.ClDataset(renders, data, window=64)
        assert len(test_set) > 0

        loader = DataLoader(test_set, batch_size=2, shuffle=False)
        for (window, label) in loader:
            assert tuple(window.size()) in [(1, 3, 64, 64), (2, 3, 64, 64)]
            assert tuple(label.size()) in [(1, ), (2, )]