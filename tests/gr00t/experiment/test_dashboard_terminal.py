"""Alternate-screen contracts for small SSH/Zellij panes; no training or CUDA."""

import errno
import io
import os
import pty
import re
import subprocess
import sys
from types import SimpleNamespace

from gr00t.experiment.console_logging import DetailedConsoleStream, copy_terminal_size
from gr00t.experiment.training_dashboard import TrainingDashboardCallback
import pytest
from rich.console import Console


def test_live_uses_one_alternate_screen_and_step_refresh(monkeypatch):
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr(sys, "stdout", output)
    dashboard = TrainingDashboardCallback(True, refresh_seconds=100)
    args = SimpleNamespace(process_index=0)
    state = SimpleNamespace(global_step=0, max_steps=40)
    dashboard.on_train_begin(args, state, None)
    try:
        assert dashboard._live._screen
        assert dashboard._live.vertical_overflow == "crop"
        for step in range(1, 31):
            state.global_step = step
            dashboard.on_step_end(args, state, None)
        assert dashboard._progress.tasks[0].completed == 30
        assert "loss/total" not in dashboard._metrics  # Never synthesize missing measurements.
        assert "N/A" in output.getvalue()
    finally:
        dashboard.on_train_end(args, state, None)
    stream = output.getvalue()
    assert stream.count("\x1b[?1049h") == 1
    assert stream.count("\x1b[?1049l") == 1
    before, rest = stream.split("\x1b[?1049h")
    frames, after = rest.split("\x1b[?1049l")
    assert "Florence" not in before + after  # No frames written into primary scrollback.
    assert frames.count("Florence") >= 30
    assert frames.count("\x1b[H") >= 30  # Replace from home, rather than append below frames.
    assert "\x1b[?25h" in stream  # Cursor restored.


@pytest.mark.parametrize("size", [(120, 48), (95, 24), (55, 18), (35, 10)])
def test_resize_adapts_layout_and_bounds_height(monkeypatch, size):
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr(sys, "stdout", output)
    dashboard = TrainingDashboardCallback(True, diagnostics_enabled=True)
    dashboard.on_train_begin(
        SimpleNamespace(process_index=0), SimpleNamespace(global_step=0, max_steps=10), None
    )
    try:
        dashboard._console.size = size
        renderable = dashboard._render()
        # Measure the chosen layout without Live's final safety crop.
        console = Console(file=io.StringIO(), width=size[0], height=size[1])
        lines = console.render_lines(renderable, console.options, pad=False)
        assert len(lines) <= size[1]
        dashboard._refresh(force=True)
    finally:
        dashboard._stop_live()


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_pty_resize_and_exception_restores_terminal(exception):
    # Real PTY: mimics inherited xterm stdout through the Hydra relay, with Zellij set.
    master, slave = pty.openpty()
    outer_master, outer_slave = pty.openpty()
    process = None
    try:
        import fcntl
        import struct
        import termios

        fcntl.ioctl(outer_slave, termios.TIOCSWINSZ, struct.pack("HHHH", 18, 60, 0, 0))
        assert copy_terminal_size(outer_slave, slave)
        assert os.get_terminal_size(slave) == (60, 18)
        code = """
import os
import sys
from types import SimpleNamespace
from gr00t.experiment.training_dashboard import TrainingDashboardCallback
ui = TrainingDashboardCallback(True)
s = SimpleNamespace(global_step=0, max_steps=20)
ui.on_train_begin(SimpleNamespace(process_index=0), s, None)
try:
    for i in range(10):
        s.global_step = i
        ui.on_step_end(None, s, None)
    print("READY_RESIZE", flush=True)
    sys.stdin.readline()
    assert os.get_terminal_size(1) == (100, 30)
    ui._refresh(force=True)
    raise EXCEPTION("simulated stop")
except EXCEPTION as exc:
    assert str(exc) == "simulated stop"
finally:
    ui._stop_live()
print("NORMAL_SCREEN_RESTORED", flush=True)
""".replace("EXCEPTION", exception.__name__)
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=slave,
            stderr=slave,
            env={**os.environ, "TERM": "xterm-256color", "ZELLIJ": "1"},
        )
        os.close(slave)
        slave = None
        chunks = []
        resized = False
        while True:
            import select

            assert select.select([master], [], [], 30)[0], "PTY child stalled"
            try:
                chunk = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)
            if not resized and b"READY_RESIZE" in b"".join(chunks):
                fcntl.ioctl(outer_slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
                assert copy_terminal_size(outer_slave, master)
                process.stdin.write(b"continue\n")
                process.stdin.flush()
                resized = True
        assert resized
        assert process.wait(timeout=30) == 0
        stream = b"".join(chunks)
        assert stream.count(b"\x1b[?1049h") == 1
        assert stream.count(b"\x1b[?1049l") == 1
        assert stream.rfind(b"\x1b[?1049l") < stream.index(b"NORMAL_SCREEN_RESTORED")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        for fd in (master, slave, outer_master, outer_slave):
            if fd is not None:
                os.close(fd)


def test_rank_noise_file_preserved_and_nontty_no_escape(monkeypatch):
    console, detail = io.StringIO(), io.StringIO()
    stream = DetailedConsoleStream(console, detail, quiet=True)
    stream.write("rank 2 ordinary stdout\n")
    stream.write("ERROR checkpoint failed")
    stream.write("\n")
    assert "ordinary" in detail.getvalue() and "ordinary" not in console.getvalue()
    assert "ERROR checkpoint failed\n" in console.getvalue()
    monkeypatch.setattr(sys, "stdout", console)
    dashboard = TrainingDashboardCallback(True)
    args, state = SimpleNamespace(process_index=0), SimpleNamespace(global_step=16, max_steps=20)
    dashboard.on_train_begin(args, state, None)
    dashboard.on_log(args, state, None, logs={"loss/total": 1.2})
    dashboard.on_train_end(args, state, None)
    assert "Step 16" in console.getvalue()
    assert not re.search(r"\x1b\[", console.getvalue())
