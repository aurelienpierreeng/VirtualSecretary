# Getting started

The Virtual Secretary has 3 parts:

1. the Python 3.11 interpreter, with all module dependencies, provided as an all-in-one AppImage package,
2. the Virtual Secretary application source code, located in the `src/` folder of the source code,
3. the user filters, typically installed in a `config/` folder.

The AppImage is heavy (350 MB) but will not require frequent updates and provides linear algebra, language processing and machine learning modules used in Virtual Secretary core and that can be reused for your own filters. It prevents you from having to resolve Python package conflicts between the versions installed on your computer or server, and doesn't need virtual environments, so it's pretty much plug-and-play.

The actual Virtual Secretary app is a 600 kB folder of source code that will need to be updated in the future.

## Getting the code

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

### AppImage

I provide [an AppImage package](https://github.com/aurelienpierreeng/VirtualSecretary/releases/download/0.0.0/VirtualSecretary-x86_64.AppImage) for Linux which contains all the necessary dependencies to run the Virtual Secretary. To download and enable it:

```bash
wget https://github.com/aurelienpierreeng/VirtualSecretary/releases/download/0.0.0/VirtualSecretary-x86_64.AppImage
chmod +x VirtualSecretary-x86_64.AppImage
```

The AppImage can be used on Linux desktops and servers, even on shared hosting and in Docker containers. However, in that case, FUSE might not be installed or even forbidden to run for safety, so you will need to extract the content of the AppImage:

```bash
./VirtualSecretary-x86_64.AppImage --appimage-extract
mv squashfs-root appimage
```

From there, we will found the Python runtime in an `appimage/` folder, in the current directory.

### Manual installation

Install [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date) and ensure you have at least Python 3.10 (3.11 is recommended). If you can't install Python 3.10 on your system, resume to the AppImage approach above. Once this is done and you grabbed the source code at the previous step, to install all dependencies at once, run from the `VirtualSecretary/` folder:

```bash
pip -r recipe/requirements.txt
```

Alternatively, use:

```bash
python3 -m pip -r recipe/requirements.txt
```

## Running the filters

Provided you have filters saved in a `VirtualSecretary/config/` folder, to run the Virtual Secretary from a default Python installation, use :

```bash
python VirtualSecretary/src/main.py VirtualSecretary/config process
```

To run it from the non-extracted AppImage (on systems supporting FUSE):

```bash
./VirtualSecretary-x86_64.AppImage path/to/src/main.py path/to/config process
```

To run it from the extracted AppImage (as shown above — assuming you extracted it to `appimage/`), on systems not supporting FUSE:

```bash
./appimage/AppRun virtualsecretary/src/main.py virtualsecretary/config/  process
```

## Creating a Cron job

You can run your filters every `n` minutes, on Linux desktop and servers, using a Cron job. If you have several devices getting emails, for example, it may be a good idea to run them on your email servers every 10-30 minutes, so all your devices get a clean input.

```bash
contab -e
*/10 * * * * python gazette/virtual-secretary/src/main.py
```
