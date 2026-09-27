from __future__ import annotations

import logging
from .agent import Agent
from .config import Config


def main() -> None:
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(Agent(config).run())


if __name__ == "__main__":
    main()
