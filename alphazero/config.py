from pathlib import Path
from omegaconf import OmegaConf, DictConfig

DEFAULT_CONF = Path(__file__).parent.parent / "conf" / "default.yaml"


def load_config(*overrides: str, base: str = None) -> DictConfig:
    base_path = base or str(DEFAULT_CONF)
    cfg = OmegaConf.load(base_path)

    files = []
    dotlist = []
    for o in overrides:
        if "=" in o and not o.endswith((".yaml", ".yml")):
            dotlist.append(o)
        else:
            files.append(o)

    for path in files:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))

    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    return cfg
