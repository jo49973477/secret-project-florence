"""Invoke the external public CLI with graceful POSIX termination.

trainctl 0.2.0 uses asyncio.run's SIGINT cancellation to reap its separate child
process group. Translate SIGTERM into that same path until upstream installs a
SIGTERM handler. This module owns no lifecycle events or supervision logic.
"""

import os
import signal


def main():
    from trainctl.cli import main as trainctl_main

    previous = signal.signal(signal.SIGTERM, lambda *_: os.kill(os.getpid(), signal.SIGINT))
    try:
        trainctl_main()
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
