if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import argparse
    import os
    import snook.data as sd
    

    BALLS  = [f"resources/fbx/ball_{color}.fbx" for color in sd.COLORS]
    CUE    = "resources/fbx/cue.fbx"
    POOL   = "resources/fbx/pool.fbx"
    HDRI   = "resources/hdri"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int,   help="# training samples")
    parser.add_argument("--valid", type=int,   help="# validation samples")
    parser.add_argument("--test",  type=int,   help="# testing samples")
    parser.add_argument("--dest",  type=str,   help="destination directory")
    parser.add_argument("--tile",  type=float, help="tile size percent")
    args = parser.parse_args()


    scene = None
    def generate(name: str, path: str, samples: int) -> None:
        global args, scene
        
        if os.path.isdir(path):
            return
        
        if scene is None:
            size = 512
            tile = int(size * min(max(args.tile, 0), 1))

            scene = sd.Scene(
                sd.cFiles(BALLS, CUE, POOL, HDRI),
                sd.cTable((2.07793, 1.03677), (0.25, 0.20), 1.70342),
                sd.cDistances(0.1047, 0.154, 1.5, (10.0, 20.0)),
                render = sd.cRender((size,) * 2, (tile,) * 2, 64, False, True)
            )
        
        renders = os.path.join(path, "renders")
        data = os.path.join(path, "data")
        
        os.makedirs(renders, exist_ok=True)
        os.makedirs(data, exist_ok=True)
        
        for i in tqdm(range(samples), desc=name):
            scene.sample()
            scene.render(f"{renders}/{i}.png")
            scene.register(f"{data}/{i}.txt")


    generate("Train", os.path.join(args.dest, "train"), args.train)
    generate("Valid", os.path.join(args.dest, "valid"), args.valid)
    generate("Test",  os.path.join(args.dest, "test"),  args.test)