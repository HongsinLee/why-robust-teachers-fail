import sys
import time


class ProgressBar:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start_time = None

    def prog(self, i, max_iter, epoch, loss):
        if not self.verbose:
            return

        if i == 0:
            self.start_time = time.time()

        progress = (i + 1) / max_iter
        bar_len = 30
        filled_len = int(bar_len * progress)
        bar = "=" * filled_len + "." * (bar_len - filled_len)

        elapsed = time.time() - self.start_time
        it_per_sec = (i + 1) / max(elapsed, 1e-8)

        msg = (
            f"\rEpoch {epoch} "
            f"[{bar}] "
            f"{i + 1}/{max_iter} "
            f"loss={loss:.4f} "
            f"it/s={it_per_sec:.2f}"
        )

        print(msg, file=sys.stderr, end="", flush=True)

        if i + 1 == max_iter:
            print("", file=sys.stderr)


def progress_bar(i, max_iter, epoch, loss):
    global static_bar

    if i == 0:
        static_bar = ProgressBar()

    static_bar.prog(i, max_iter, epoch, loss)