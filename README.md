# RNNs-learn-robust-policies

## Quickstart
- Create a virtual environment and sync deps: `uv sync`.
- Set a persistent config dir (optional but recommended): `export RLRMP_CONFIG_DIR=~/rlrmp-config` and create minimal overrides (see Configuration Tips below).
- Run training and analysis entrypoints with `uv run`, using the current
  experiment-specific scripts under `scripts/`.

- Set `RLRMP_CONFIG_DIR` to point to your local overrides directory for experiment/analysis defaults.
- Default paths base is `paths.yml: base: /tmp/rlrmp`, which is ephemeral. To persist DB/models/figures, create `~/rlrmp-config/paths.yml` like:

  ```yaml
  # ~/rlrmp-config/paths.yml
  base: ~/rlrmp-data
  db: db
  models: models
  figures: figures
  figures_dump: figures_dump
  cache: cache
  logs: logs
  ```

- Place training/analysis YAML overrides under `$RLRMP_CONFIG_DIR/training` and `$RLRMP_CONFIG_DIR/analysis` mirroring the package layout. For example: `$RLRMP_CONFIG_DIR/analysis/part2/plant_perts.yml`.

### Troubleshooting

#### Kaleido error

Such as:

```
ValueError: Failed to start Kaleido subprocess. Error stream:

/Users/mll/main/10 Projects/10 PhD/41 RNNs learn robust policies/.venv/lib/python3.12/site-packages/kaleido/executable/kaleido: line 4: cd: /Users/mll/main/10: No such file or directory
```

Edit this file (approximate location wrt. virtual environment folder):

`.venv/lib/python/site-packages/kaleido/executable/kaleido`

Change `cd $DIR` to `cd "$DIR"`, and `$@` to `"$@"`.

This error tends to arise when there are spaces in the path to the directory in which you save your
analysis figures in raster formats (e.g. PNG).

This issue was eliminated 

## Configuration

Set environment variable `RLRMP_CONFIG_DIR` to the path of your configuration directory.

Then any files in that directory will be loaded instead of the respective files in 
the subpackage `config`, *when loading config files by name*. 
This way you can change the default hyperparameters for an entire part of the experiment.

Unless otherwise indicated, *default* config files, including the ones you modify in your own config directory, need to be complete; i.e. you should not omit any of the keys found in the defaults included in the subpackage `config`.

(Show how to copy one of the default config files to your own directory)
e.g. `~/.config/rlrmp`.

When using the script `train` to train models on tasks, or the script `run_analysis` to run analyses,
the config is passed as a path to a YAML file. 
These config files are allowed to be incomplete, and in fact it may be convenient to use them to specify 
only one or a few parameters that differ from the defaults.
The scripts use the partial configs to update the defaults.

In particular, consider the `load` subtree found in config files for the analysis notebooks,
which indicates the hyperparameters of the model(s)/training run(s) to load for analysis. We should only include those 
parameters that uniquely identify the model(s) we want to load, so we may omit some of the hyperparameters
that must appear in the default *training* configs.

If you have a space or other special character in your venv path, it may be necessary to modify your kaleido executable script slightly, for Plotly image export to work: https://github.com/plotly/Kaleido/issues/57#issuecomment-1024462647
