if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from utils import plot_render_mask_heatmap

    import argparse
    import os
    import snook.data as sd
    

    BALLS  = [f"resources/fbx/ball_{color}.fbx" for color in sd.COLORS]
    CUE    = "resources/fbx/cue.fbx"
    POOL   = "resources/fbx/pool.fbx"
    HDRI   = "resources/hdri"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, helper="# training samples")
    parser.add_argument("--valid", type=int, helper="# validation samples")
    parser.add_argument("--test",  type=int, helper="# testing samples")
    parser.add_argument("--dest",  type=str, helper="destination directory")
    args = parser.parse_args()


    scene = None
    def generate(name: str, path: str, samples: int) -> None:
        global scene
        
        if os.path.isdir(path):
            return
        
        if scene is None:
            scene = sd.Scene(
                sd.cFiles(BALLS, CUE, POOL, HDRI),
                sd.cTable((2.07793, 1.03677), (0.25, 0.20), 1.70342),
                sd.cDistances(0.1047, 0.154, 1.5, (10.0, 20.0)),
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