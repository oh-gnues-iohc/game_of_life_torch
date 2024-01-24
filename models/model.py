import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

size = 10
n_frames = 240
full_size = (1, 1, size, size)

env = torch.tensor([[[[0., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]]).float()

class PaddingLayer(nn.Module):
    def __init__(self, size):
        super(PaddingLayer, self).__init__()
        self.size = size

    def forward(self, x):
        return F.pad(x, (1,1,1,1), mode='circular')

def kernel(size):
    kernel = torch.ones(1, 1, 3, 3)
    kernel[0, 0, 1, 1] = 0  # Don't count the cell itself
    return kernel

# Define the model
class GameOfLifeModel(nn.Module):
    def __init__(self, size):
        super(GameOfLifeModel, self).__init__()
        self.padding_layer = PaddingLayer(size)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
        self.conv.weight = nn.Parameter(kernel(size), requires_grad=False)

    def forward(self, x):
        x = self.padding_layer(x)
        x = self.conv(x)
        return x

if __name__ == "__main__":
    model = GameOfLifeModel(size)

    frames = []
    for i in range(n_frames):
        neighbours = model(env)
        env = torch.where(
            ((env > 0) & torch.isin(neighbours, torch.tensor([2, 3]))) | ((env == 0) & (neighbours == 3)),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32)
        )

        frames.append(env.squeeze())

    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(xlim=(0, size), ylim=(0, size))
    render = plt.imshow(frames[0].numpy(), interpolation="none", cmap="binary")

    def animate(i: int):
        render.set_array(frames[i].numpy())
        return [render]

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=30, blit=True)
    plt.axis("off")
    plt.gca().invert_yaxis()
    anim.save("glider.gif", fps=30)
    plt.show()
