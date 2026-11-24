"""Execute one notebook from a fresh IPython process and save its outputs."""
import base64
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import matplotlib
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support


def mime_data(data):
    return {
        key: base64.b64encode(value).decode("ascii") if isinstance(value, bytes) else value
        for key, value in data.items()
    }


def main(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    os.chdir(path.parent)
    matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    shell = InteractiveShell.instance()
    configure_inline_support(shell, "inline")
    shell.user_ns["display"] = display
    shell.history_manager.enabled = False
    for i, cell in enumerate((c for c in notebook["cells"] if c["cell_type"] == "code"), 1):
        start = perf_counter()
        code = "".join(cell["source"])
        with capture_output() as captured:
            result = shell.run_cell(code, store_history=True)
        if not result.success:
            print(captured.stdout, file=sys.stderr)
            print(captured.stderr, file=sys.stderr)
            raise RuntimeError(f"{path.name}: erro na célula {i}") from (
                result.error_before_exec or result.error_in_exec
            )
        cell["execution_count"] = i
        cell["outputs"] = []
        cell["metadata"].pop("execution", None)
        for stream in ("stdout", "stderr"):
            value = getattr(captured, stream)
            if value:
                cell["outputs"].append(
                    {"output_type": "stream", "name": stream, "text": value.splitlines(keepends=True)}
                )
        for rich in captured.outputs:
            cell["outputs"].append(
                {"output_type": "display_data", "data": mime_data(rich.data), "metadata": rich.metadata}
            )
        print(f"{path.name} · célula {i} OK ({perf_counter() - start:.2f} s)", flush=True)
    notebook["metadata"]["language_info"]["version"] = platform.python_version()
    notebook["metadata"]["kernelspec"]["display_name"] = "Python 3 (ipykernel)"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
