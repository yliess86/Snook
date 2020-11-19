from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
plt.style.use('fivethirtyeight')
plt.ion()


def plot_history(train: List[float], valid: List[float]) -> None:
    fig = plt.figure(figsize=(3 * 4, 4), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    
    line1, *_ = ax.plot(train, color="b")
    line2, *_ = ax.plot(valid, color="g")
    ax.legend(handles=[line1, line2], labels=["train", "valid"])

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
    ax1.title.set_text("Source Render")

    heatmap = heatmap.detach().cpu().numpy()
    ax2.imshow(heatmap)
    ax2.set_axis_off()
    ax2.title.set_text("Heatmap Ground Truth")

    prediction = prediction.detach().cpu().numpy()
    ax3.imshow(prediction)
    ax3.set_axis_off()
    ax3.title.set_text("Heatmap Prediction")

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig.canvas.draw()
    plt.show()