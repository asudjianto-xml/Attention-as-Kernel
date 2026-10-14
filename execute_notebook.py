"""Execute the notebook using this Python environment, from any working directory."""
import argparse
import os
from pathlib import Path
import sys

from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Save figures, results, and executed notebook here (default: ./outputs).')
    parser.add_argument('--skip-housing', action='store_true', help='Run only the offline synthetic examples.')
    args = parser.parse_args()
    output = (args.output_dir or Path(os.environ.get('ATTENTION_TUTORIAL_OUTPUT_DIR', 'outputs'))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).resolve().with_name('attention_tutorial.ipynb')
    notebook = nbformat.read(path, as_version=4)
    env = os.environ.copy()
    env['ATTENTION_TUTORIAL_OUTPUT_DIR'] = str(output)
    if args.skip_housing:
        env['ATTENTION_TUTORIAL_SKIP_HOUSING'] = '1'
    # Use the interpreter running this script, even if a user-level kernelspec
    # points at a different environment.
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=300,
                            resources={'metadata': {'path': str(Path.cwd())}})
    try:
        client.execute(env=env)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    destination = output / 'attention_tutorial.executed.ipynb'
    nbformat.write(notebook, destination)
    print(f'Executed notebook saved to {destination}')


if __name__ == '__main__':
    main()
