import tomllib
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
from einops import rearrange

from common.display import config_as_token
from common.loading import load_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=r"configs/cifar100.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
        # logger.info(f"Config settings.....{config_as_token(config)}")

    model = load_model(
        config["model"], (3, 32, 32)
    )  # hardcoded input shape for CIFAR100

    # Assumes that there is an activation layer after each convolutional layer
    n_plots = len([i.shape for i in model.parameters()]) // 2
    fig, ax = plt.subplots(3, n_plots // 3)
    ax = ax.ravel()
    for idx, p in enumerate(model.parameters()):
        if p.dim() < 4:
            continue
        k, c, h, w = p.shape
        p = rearrange(p, "out_c in_c h w -> (h out_c) (in_c w)")
        ax[idx // 2].set_yticks(
            [i for i in range(0, p.shape[0], 3)], ["" for i in range(0, p.shape[0], 3)]
        )
        ax[idx // 2].set_xticks(
            [i for i in range(0, p.shape[1], 3)], ["" for i in range(0, p.shape[1], 3)]
        )
        ax[idx // 2].imshow(
            p.cpu().detach().numpy() ** 2, cmap="plasma", vmin=0, vmax=0.1
        )
    plt.show()
