# Main script

## Basic use

The `src/main.py` script is the entry point of the application. It will detect all filters declared in the config directory and run them in sequence, while keeping logfiles for user traceability as well as to avoid filtering the same objects more than once (in case a false-positive was manually reverted, avoid reprocessing it later).

When you grab the source code from Github (see [Install](../starting/1-install.md)), you get a `VirtualSecretary` directory. You can add a `config` folder in it, to obtain the following tree:

- `VirtualSecretary/`:
    - `config/` __(add it)__
    - `data/` (default)
    - `models/` (default)
    - `src/` (default)
        - `core/` (default)
        - `protocols/` (default)
        - `tests/` (default)
        - `user_scripts/` (default)


The `config`, `models`, `data` and `src/user_scripts` folders are ignored by Git so you can write in them with no fear that the next `git pull` will override them. The [configuration details](../starting/2-configure.md) and [base filters](../starting/4-example-filters.md) are covered in a specific page.

Once you have a `config` directory and some filters to process, you can execute:

```bash
cd VirtualSecretary
python src/main.py config/ process
```

## General use

```bash
python src/main.py CONFIG_DIR MODE
```

- `CONFIG_DIR`: the top directory where all configuration files are hosted
- `MODE` is either:
    - `process`: to run the automation filters (which names starts by `01-imap-`, `02-caldav-`, etc),
    - `learn`: to run the machine-learning filters (which names starts by `LEARN-`).

## Options

Options need to be added __after__ `python src/main.py config/ process`. The available options are :

- `-h`: display the help of the script in the terminal (showing this list of options and their meaning),
- `-s <SUBDIR>`, `--single <SUBDIR>`: process only the specified `<SUBDIR>` sub-directory of the configuration directory (replace `<SUBDIR>` by the real sub-directory name). This is useful to debug filters on a single mail account.
- `-n <INT>`, `--number <INT>`: globally override the number of items to process defined in `settings.ini` with the `<INT>` value. This is useful when running the application for a first time, when you need to catch on.
- `--server`: enable the server mode, which adds a sleeping time between each element filtering, as to reduce system load and allow more breathing room to other running processes. This is useful if you keep the application running in background all the time, as it reduces the average CPU usage below 2% on Intel Xeon and matters if you run the application on small servers or on server instances where CPU load spikes are billed extra (like Amazon Elastic Cloud).
- `-f`, `--force`: globally force reprocessing already-processed items. Normally, items are filtered only once (unless specified otherwise in filter files) and never filtered again on future runs. This is helpful if a filter didn't work as expected, and you modified it.
