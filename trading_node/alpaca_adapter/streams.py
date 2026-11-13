import threading
from contextlib import suppress
from typing import Protocol

STOP_TIMEOUT_SECONDS = 5.0


class BlockingStream(Protocol):
    def run(self) -> None: ...

    def stop(self) -> None: ...


class StreamThread:
    """Runs a blocking alpaca-py stream in a daemon thread until stopped."""

    def __init__(self, stream: BlockingStream, name: str) -> None:
        self._stream = stream
        self._thread = threading.Thread(target=stream.run, name=name, daemon=True)

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        if not self._thread.is_alive() and self._thread.ident is None:
            self._thread.start()

    def stop(self) -> None:
        """Stop the stream; alpaca-py raises AttributeError if it never reached its event loop."""
        if not self._thread.is_alive():
            return
        with suppress(AttributeError):
            self._stream.stop()
        self._thread.join(STOP_TIMEOUT_SECONDS)
