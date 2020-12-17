if __name__ == "__main__":
    from tqdm import tqdm
    from torch.cuda.amp import autocast, GradScaler
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torchvision.transforms import ColorJitter, ToTensor

    import argparse
    import numpy as np
    import os
    import snook.data as sd
    import snook.model as sm
    import torch


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, help="training epochs")
    parser.add_argument("--refine",     type=int, help="refining epochs")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--n_workers",  type=int, help="# worker threads")
    parser.add_argument("--dataset",    type=str, help="dataset directory")
    parser.add_argument("--save",       type=str, help="model save path")
    args = parser.parse_args()

    transforms = [
        ColorJitter(0.2, 0.2, 0.2, 0.1),
        ToTensor(),
        sd.RandomLinearMotionBlur(length=(3, 15), angle=(0, 360), p=0.5),
    ]

    datasets = {
        "train": sd.ReMaHeDataset(
            os.path.join(args.dataset, "train/renders"),
            os.path.join(args.dataset, "train/data"),
            spread=4.0,
            transforms=transforms,
        ),
        "valid": sd.ReMaHeDataset(
            os.path.join(args.dataset, "valid/renders"),
            os.path.join(args.dataset, "valid/data"),
            spread=4.0,
        ),
        "test": sd.ReMaHeDataset(
            os.path.join(args.dataset, "test/renders"),
            os.path.join(args.dataset, "test/data"),
            spread=4.0,
        ),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            pin_memory=True,
        ),
    }


    layers = [sm.Layer(16, 24, 1), sm.Layer(24, 32, 6), sm.Layer(32, 64, 6)]
    model = sm.AutoEncoder(layers, 3, 1, scale=0.4).cuda()
    criterion = sm.AdaptiveWingLoss().cuda()
    optim = AdamW(model.parameters())
    scaler = GradScaler()
    def step(
        name: str,
        loader: DataLoader,
        spread: float,
        is_train: bool = True,
    ) -> float:
        global model, criterion, optim, scaler
        model = model.train() if is_train else model.eval()
        
        total_loss = 0.0
        pbar = tqdm(loader, name)
        for render, mask, heatmap in pbar:
            render, mask, heatmap = render.cuda(), mask.cuda(), heatmap.cuda()
            
            if is_train:
                optim.zero_grad()
            
            with autocast():
                loss = criterion(model(render).squeeze(1), heatmap)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / len(loader), spread=spread)
        
        return total_loss / len(loader)


    spread_range = 1000, 16
    history = {"train": [], "valid": []}
    for epoch in tqdm(range(args.epochs + args.refine), desc="Epoch"):
        t = min(epoch, args.epochs) / args.epochs
        spread = max(spread_range) - t * abs(np.subtract(*spread_range))
        loaders["train"].dataset.spread = spread
        loaders["valid"].dataset.spread = spread
        
        step("Train", loaders["train"], spread, is_train=True)
        with torch.no_grad():
            step("Valid", loaders["valid"], spread, is_train=False)
            
    with torch.no_grad():
        step("Test", loaders["test"], spread, is_train=False)

    fake_input = torch.rand((1, 3, 512, 512))
    torch.jit.save(torch.jit.trace(model.cpu(), fake_input), args.save)