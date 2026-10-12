"""Execute the actual notebook cells, record outputs, and save experiment metrics."""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import types
import uuid

import torch

from notebook_tools import ROOT


class Tee(io.StringIO):
    def write(self, text):
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        return super().write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercises", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument("--ex3-epochs", type=int, default=3)
    parser.add_argument("--ex4-epochs", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.ex3_epochs, args.ex4_epochs, args.threads) < 1:
        parser.error("Epoch and thread counts must be positive")
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    for exercise in args.exercises:
        path = ROOT / "src" / f"ex{exercise}.ipynb"
        notebook = json.loads(path.read_text())
        module = types.ModuleType(f"hw1_run_ex{exercise}")
        sys.modules[module.__name__] = module
        print(f"Executing {path.name}", flush=True)
        count = 0
        for cell in notebook["cells"]:
            cell.setdefault("id", uuid.uuid4().hex[:8])
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if exercise == 3 and "train_losses = train_classifier" in source:
                source = source.replace("epochs=3", f"epochs={args.ex3_epochs}")
                cell["source"] = source.splitlines(keepends=True)
            if exercise == 4 and "cfg = TrainConfig(" in source:
                source = source.replace("epochs=5", f"epochs={args.ex4_epochs}")
                source = re.sub(r"seeds = \[[^\n]*\]", f"seeds = {args.seeds!r}", source)
                # Record the actual configuration used alongside the measured outputs.
                cell["source"] = source.splitlines(keepends=True)
            count += 1
            output = Tee()
            with contextlib.redirect_stdout(output):
                exec(compile(source, f"{path.name}:cell{count}", "exec"), module.__dict__)
            cell["execution_count"] = count
            cell["outputs"] = ([{"output_type": "stream", "name": "stdout", "text": output.getvalue().splitlines(keepends=True)}]
                               if output.getvalue() else [])
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        print(f"Saved {path.name}", flush=True)


if __name__ == "__main__":
    main()
