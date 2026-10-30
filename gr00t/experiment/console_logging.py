"""Preserve worker streams while removing only known repetitive console records."""

import re


class DetailedConsoleStream:
    def __init__(self, stream, detail, quiet=False):
        self.stream, self.detail = stream, detail
        self.quiet = quiet
        self._drop_newline = False
        self._keep_newline = False

    def write(self, text):
        self.detail.write(text)
        self.detail.flush()
        if self._keep_newline and text == "\n":
            self._keep_newline = False
            self.stream.write(text)
            return len(text)
        self._keep_newline = False
        if self._drop_newline and text == "\n":
            self._drop_newline = False
            return len(text)
        self._drop_newline = False
        for line in text.splitlines(keepends=True):
            important = re.search(
                r"warning|error|exception|traceback|out of memory|failed|failure|critical",
                line,
                re.I,
            )
            noisy = re.search(r"Rank \d+, Worker \d+: (Caching shard|Wait for shard)", line)
            noisy = (
                noisy
                or "Model-load trace " in line
                or "Model-load memory [" in line
                or "checkpoint-memory phase=" in line
            )
            if important or (not noisy and not self.quiet):
                self.stream.write(line)
                self._keep_newline = not line.endswith("\n")
            elif not line.endswith("\n"):
                self._drop_newline = True
        return len(text)

    def flush(self):
        self.detail.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def copy_terminal_size(source_fd, target_fd):
    """Keep the worker PTY's geometry synchronized with SSH/Zellij's outer PTY."""
    import fcntl
    import termios

    try:
        size = fcntl.ioctl(source_fd, termios.TIOCGWINSZ, b"\0" * 8)
        fcntl.ioctl(target_fd, termios.TIOCSWINSZ, size)
    except OSError:
        return False
    return True
