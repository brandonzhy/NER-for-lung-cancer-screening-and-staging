import torch
import os


cache_root = os.path.expanduser(os.path.join("~", ".NLPer"))


device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

from . import data
from . import models
from . import trainers
from . import nn

import logging.config


logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "NLPer": {"handlers": ["console"], "level": "INFO", "propagate": False}
        },
        "root": {"handlers": ["console"], "level": "WARNING"},
    }
)

logger = logging.getLogger("NLPer")
