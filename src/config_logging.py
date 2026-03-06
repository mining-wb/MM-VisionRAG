# ====== 日志配置 ======
# 控制台 + app.log，供各模块 import 后使用 getLogger(__name__)

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = LOG_DIR / "app.log"


def setup_logging(level: int = logging.INFO) -> None:
    """配置根 logger：控制台 + 文件 app.log。"""
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
