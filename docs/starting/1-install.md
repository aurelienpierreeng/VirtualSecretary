# Install

The Virtual Secretary has 4 parts:

1. the Python 3.11 interpreter (aka __the runtime__),
2. the Python packages upon which it depends (aka __the libraries__),
2. the Virtual Secretary application source code (aka __the application__), located in the `src/` folder of the source code,
3. the user filters (aka __the implementation__), typically installed in a `config/` folder.

The Python interpreter is pretty standard in Linux distributions and easy to install on Windows and Mac. However, Linux distributions may have older versions than 3.11. If you plan on using the Virtual Secretary on a shared hosting server, you will not have access to a shell giving you rights to install or update Python. For those cases where you can't install or update the Python interpreter to version 3.11, [download the Python 3.11 AppImage](https://github.com/niess/python-appimage/releases/download/python3.11/python3.11.4-cp311-cp311-manylinux_2_28_x86_64.AppImage). See below how to deploy Python 3.11 to a server where you don't have `sudo` rights.

## Getting the source code

The easiest way is to get and update the source code through Git ([installation instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)).

Once Git is installed, in a terminal, run:

```bash
git clone https://github.com/aurelienpierreeng/VirtualSecretary.git
cd VirtualSecretary
```

To update in the future, run:

```bash
cd VirtualSecretary
git pull
```

## Preparing the runtime

### Python interpreter

If you have access to a package manager and you are able to simply install [Python 3.11](https://www.python.org/downloads/release/python-3114/), do it. Otherwise, here is how to deploy the AppImage on a server:

0. We assume here you are already in `VirtualSecretary` folder, after the `cd VirtualSecretary` command at the previous step.
1. Download the AppImage package:
```bash
wget https://github.com/niess/python-appimage/releases/download/python3.11/python3.11.4-cp311-cp311-manylinux2014_x86_64.AppImage
```

2. Rename it `python3.11.AppImage` for convenience:
```bash
mv python3.11.4-cp311-cp311-manylinux2014_x86_64.AppImage VirtualSecretary/python3.11.AppImage
```

3. Give it execution permissions:
```bash
chmod +x python3.11.AppImage
```

4. Run the test command:
```bash
./python3.11.AppImage --version
```
If it returns:
```bash
dlopen(): error loading libfuse.so.2
AppImages require FUSE to run.
```
then you need to extract the AppImage because the system can't do it. Luckily, the AppImage package is self-extractable, so just run:
```bash
./python3.11.AppImage --appimage-extract && mv squashfs-root python3.11
```

From there, to execute a Python script (`.py`), you have to use one of the following methods:

1. `python3.11 script.py` if you have multiple versions of Python installed,
2. `python script.py` if you have only Python 3.11 installed,
3. `./python3.11.AppImage script.py` from within the `VirtualSecretary` folder if you can use the non-extracted AppImage
4. `python3.11/AppRun script.py` from within the `VirtualSecretary` folder if you had to extract the AppImage

In the rest of this document, we will write `PYTHON` as a generic way of calling the interpreter that will need to be replaced by one of the above commands depending on your case.

### Install the dependencies

From the `VirtualSecretary/` folder, run:
```bash
PYTHON -m pip install -r recipe/requirements.txt
```

!!! note

    This installation procedure was tested on Ubuntu Server 20.04, on Fedora Desktop 37 and 38, and on a CPanel-based shared hosting (Debian).
