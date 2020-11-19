from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
plt.style.use('fivethirtyeight')
plt.ion()


def plot_render_mask_heatmap(
    render: torch.Tensor, mask: torch.Tensor, heatmap: torch.Tensor,
) -> None:
    fig = plt.figure(figsize=(3 * 4, 4), facecolor="white")
    ax1, ax2, ax3 = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

    render = render.detach().cpu().permute((1, 2, 0)).numpy()
    ax1.imshow(render)
    ax1.set_axis_off()
    ax1.title.set_text("Render")

    mask = mask.detach().cpu().numpy()
    ax2.imshow(mask)
    ax2.set_axis_off()
    ax2.title.set_text("Mask")

    heatmap = heatmap.detach().cpu().numpy()
    ax3.imshow(heatmap)
    ax3.set_axis_off()
    ax3.title.set_text("Heatmap")

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig.canvas.draw()
    plt.show()


def plot_render_heatmap_prediction(
    render: torch.Tensor, heatmap: torch.Tensor, prediction: torch.Tensor,
) -> None:
    fig = plt.figure(figsize=(3 * 4, 4), facecolor="white")
    ax1, ax2, ax3 = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

    render = render.detach().cpu().permute((1, 2, 0)).numpy()
    ax1.imshow(render)
    ax1.set_axis_off()
    ax1.title.set_text("Render")

    heatmap = heatmap.detach().cpu().numpy()
    ax2.imshow(heatmap)
    ax2.set_axis_off()
    ax2.title.set_text("Heatmap")

    prediction = prediction.detach().cpu().numpy()
    ax3.imshow(prediction)
    ax3.set_axis_off()
    ax3.title.set_text("Prediction")

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig.canvas.draw()
    plt.show()


def plot_window_probabilities(
    window: torch.Tensor, probabilities: torch.Tensor,
) -> None:
    fig = plt.figure(figsize=(2 * 4, 4), facecolor="white")
    ax1, ax2 = [fig.add_subplot(1, 2, i + 1) for i in range(2)]

    window = window.detach().cpu().permute((1, 2, 0)).numpy()
    ax1.imshow(window)
    ax1.set_axis_off()
    ax1.title.set_text("Render")

    probabilities = probabilities.detach().cpu().numpy()
    best = np.max(probabilities)
    labels = ["black", "white", "yellow", "red", "cue"]
    colors = ["#008FD5" if p == best else "#7ec6e5" for p in probabilities]
    poses = range(len(probabilities))
    ax2.bar(poses, probabilities, tick_label=labels, color=colors)
    ax2.title.set_text("Probabilities")
    ax2.grid(axis="x")
    ax2.xaxis.set_tick_params(rotation=45)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig.canvas.draw()
    plt.show()


def plot_loss_history(train: List[float], valid: List[float]) -> None:
    fig = plt.figure(figsize=(4, 4), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    
    line1, *_ = ax.plot(train, color="#008FD5")
    line2, *_ = ax.plot(valid, color="#E5AE42")
    ax.legend(handles=[line1, line2], labels=["train loss", "valid loss"])

    fig.canvas.draw()
    plt.show()


def plot_acc_history(train: List[float], valid: List[float]) -> None:
    fig = plt.figure(figsize=(4, 4), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    
    line1, *_ = ax.plot(train, color="#008FD5")
    line2, *_ = ax.plot(valid, color="#E5AE42")
    ax.legend(handles=[line1, line2], labels=["train acc", "valid acc"])
    ax.set_ylim(0, 1)

    fig.canvas.draw()
    plt.show()