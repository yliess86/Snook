if __name__ == '__main__':
    from tqdm import tqdm
    from torch.cuda.amp import autocast, GradScaler
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torchvision.transforms import (
        ColorJitter,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        RandomRotation,
        ToTensor,
    )
    from typing import Tuple

    import argparse
    import numpy as np
    import os
    import snook.data as sd
    import snook.model as sm
    import torch
    import torch.nn as nn


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, help="raining epochs")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--n_workers",  type=int, help="# worker threads")
    parser.add_argument("--dataset",    type=str, help="dataset directory")
    parser.add_argument("--save",       type=str, help="model save path")
    args = parser.parse_args()

    transforms = [
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(360),
        ColorJitter(0.2, 0.2, 0.2, 0.1),
        ToTensor(),
        sd.RandomLinearMotionBlur(length=(3, 15), angle=(0, 360), p=0.5),
    ]

    datasets = {
        "train": sd.ClDataset(
            os.path.join(args.dataset, "train/renders"),
            os.path.join(args.dataset, "train/data"),
            transforms=transforms,
        ),
        "valid": sd.ClDataset(
            os.path.join(args.dataset, "valid/renders"),
            os.path.join(args.dataset, "valid/data"),
        ),
        "test": sd.ClDataset(
            os.path.join(args.dataset, "test/renders"),
            os.path.join(args.dataset, "test/data"),
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


    layers    = [
        sm.Layer(  3,  16, 3),
        sm.Layer( 16,  32, 3),
        sm.Layer( 32,  64, 3),
        sm.Layer( 64, 128, 3),
        sm.Layer(128, 256, 3),
    ]
    model = sm.Classifier(
        layers,
        hidden=512,
        n_class=len(sd.COLORS) + 1,
        scale=0.4
    ).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optim = AdamW(model.parameters())
    scaler = GradScaler()
    def step(
        name: str,
        loader: DataLoader,
        dataset: sd.ClDataset,
        is_train: bool = True,
    ) -> Tuple[float, float]:
        global model, criterion, optimizer, scaler
        model = model.train() if is_train else model.eval()
        
        total_loss, total_acc = 0.0, 0.0
        pbar = tqdm(loader, name)
        for window, label in pbar:
            window, label = window.cuda(), label.cuda()
            
            if is_train:
                optim.zero_grad()
            
            with autocast():
                logits = model(window)
                loss = criterion(logits, label)
                acc = (torch.argmax(logits, dim=1) == label).sum()
            
            loss = scaler.scale(loss)
            acc = scaler.scale(acc)

            if is_train:
                loss.backward()
                scaler.step(optim)
                
            scaler.update()
            
            total_loss += loss.item()
            total_acc += acc.item()
            pbar.set_postfix(
                loss=total_loss / len(loader),
                acc=total_acc / len(dataset),
            )
        
        return total_loss / len(loader), total_acc / len(dataset)


    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        step("Train", loaders["train"], datasets["train"], is_train=True)
        
        with torch.no_grad():
            step("Valid", loaders["valid"], datasets["valid"], is_train=False)
                
    with torch.no_grad():
        step("Test", loaders["test"], datasets["test"], is_train=False)
        
    fake_input = torch.rand(1, 3, 64, 64)
    torch.jit.save(torch.jit.trace(model.cpu(), fake_input), args.save)