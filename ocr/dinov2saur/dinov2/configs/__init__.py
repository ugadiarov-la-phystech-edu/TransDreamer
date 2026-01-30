import pathlib

from omegaconf import OmegaConf


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


dinov2_default_config = load_config("ssl_default_config")


def get_cfg_from_args(config_path):
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg
