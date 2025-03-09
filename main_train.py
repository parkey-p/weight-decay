from pathlib import Path
import logging
from datetime import datetime

if __name__ == "__main__":
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path().joinpath("runs", current_datetime + "_")
    log_path.mkdir(exist_ok=True)
    log_file = log_path.joinpath("log.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging

from argparse import ArgumentParser
import tomllib
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from common.display import config_as_token
from common.loading import load_dataset, load_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=r"configs/cifar100.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
        logger.info(f"Config settings.....{config_as_token(config)}")
    # write the config in config format for easier re-running
    with open(log_path.joinpath("config.toml"), "wb") as f:
        tomllib.dump(config, f)

    ds = load_dataset(config["data_set"])
    ds_test = load_dataset(config.get("data_set_test", {}))

    dl_kwargs = config["data_loader"]
    dl = torch.utils.data.DataLoader(ds, **dl_kwargs)
    if ds_test is not None:
        dl_test = torch.utils.data.DataLoader(
            ds_test, **config.get("data_loader_test", {})
        )
    else:
        dl_test = None

    model = load_model(config["model"], ds[0][0].size())

    summary_writer = SummaryWriter(str(log_path))
    model.fit(
        **config["train"],
        data_loader=dl,
        writer=summary_writer,
        test_data_loader=dl_test,
    )
    model.save(log_path.joinpath("model.ckpt"))
