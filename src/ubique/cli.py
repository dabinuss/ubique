from __future__ import annotations

import logging
import os

from .brain.runtime import NeurocognitiveRuntime
from .config import Config


def main() -> None:
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if os.getenv('UBIQUE_RUNTIME_VERSION', '2').strip() == '3':
        from .v3.runtime import UbiqueV3
        raise SystemExit(UbiqueV3(config).run())
    raise SystemExit(NeurocognitiveRuntime(config).run())


if __name__ == "__main__":
    main()
