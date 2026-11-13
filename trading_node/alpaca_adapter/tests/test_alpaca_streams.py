import threading

from trading_node.alpaca_adapter.streams import StreamThread


class FakeStream:
    def __init__(self, stop_error: Exception | None = None) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.runs = 0
        self.stops = 0
        self.stop_error = stop_error

    def run(self) -> None:
        self.runs += 1
        self.started.set()
        self.release.wait()

    def stop(self) -> None:
        self.stops += 1
        self.release.set()
        if self.stop_error:
            raise self.stop_error


def test_start_runs_the_stream_in_a_daemon_thread_once() -> None:
    stream = FakeStream()
    thread = StreamThread(stream, "test-stream")
    thread.start()
    thread.start()
    assert stream.started.wait(2)
    assert thread.is_running
    assert stream.runs == 1
    thread.stop()


def test_stop_stops_the_stream_and_joins_the_thread() -> None:
    stream = FakeStream()
    thread = StreamThread(stream, "test-stream")
    thread.start()
    stream.started.wait(2)
    thread.stop()
    assert stream.stops == 1
    assert not thread.is_running


def test_stop_before_start_does_nothing() -> None:
    stream = FakeStream()
    StreamThread(stream, "test-stream").stop()
    assert stream.stops == 0


def test_stop_tolerates_a_stream_that_never_reached_its_loop() -> None:
    stream = FakeStream(
        stop_error=AttributeError("'NoneType' object has no attribute 'is_running'")
    )
    thread = StreamThread(stream, "test-stream")
    thread.start()
    stream.started.wait(2)
    thread.stop()
    assert not thread.is_running


def test_a_stopped_thread_is_not_restarted() -> None:
    stream = FakeStream()
    thread = StreamThread(stream, "test-stream")
    thread.start()
    stream.started.wait(2)
    thread.stop()
    thread.start()
    assert stream.runs == 1
