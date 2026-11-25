import os
import sys


class TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self.stream, name)


def setup_run_log(experiment_dir: str, filename: str) -> str:
    log_dir = os.path.join(experiment_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_path
