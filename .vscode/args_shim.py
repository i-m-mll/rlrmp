# .vscode/args_shim.py
import os, sys, shlex, runpy

# Get extra args as a single string; allow empty.
extra = os.environ.get("EXTRA_ARGS", "")
if extra.strip():
    # Insert right after the module name + base args you configured in launch.json
    sys.argv.extend(shlex.split(extra))

# Now run your real entrypoint as a module
# Equivalent to: python -m feedbax_experiments.bin.run <your base args> [extra...]
runpy.run_module("feedbax_experiments.bin.run", run_name="__main__")