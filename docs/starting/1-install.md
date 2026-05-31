# Install

The Virtual Secretary has 4 parts:

1. the Python 3.12 interpreter (aka __the runtime__),
2. the Python packages upon which it depends (aka __the libraries__),
2. the Virtual Secretary application source code (aka __the application__), located in the `src/` folder of the source code,
3. the user filters (aka __the implementation__), typically installed in a `config/` folder.

The Python interpreter is pretty standard in Linux distributions and easy to install on Windows and Mac. However, Linux distributions may have older versions than 3.12. If you plan on using the Virtual Secretary on a shared hosting server, you will not have access to a shell giving you rights to install or update Python. For those cases where you can't install or update the Python interpreter to version 3.12, [download the Python 3.12 AppImage](https://github.com/niess/python-appimage/releases/download/python3.12/python3.12.4-cp311-cp311-manylinux_2_28_x86_64.AppImage). See below how to deploy Python 3.12 to a server where you don't have `sudo` rights.

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

## Python interpreter

If you have access to a package manager and you are able to simply install [Python 3.12](https://www.python.org/downloads/latest/python3.12/), do it. Otherwise, here is how to deploy the AppImage on a server:

0. We assume here you are already in `VirtualSecretary` folder, after the `cd VirtualSecretary` command at the previous step.
1. Download the AppImage package:
```bash
wget https://github.com/niess/python-appimage/releases/download/python3.12/python3.12.4-cp311-cp311-manylinux2014_x86_64.AppImage
```

2. Rename it `python3.12.AppImage` for convenience:
```bash
mv python3.12.4-cp311-cp311-manylinux2014_x86_64.AppImage python3.12.AppImage
```

3. Give it execution permissions:
```bash
chmod +x python3.12.AppImage
```

4. Run the test command:
```bash
./python3.12.AppImage --version
```
If it returns:
```bash
dlopen(): error loading libfuse.so.2
AppImages require FUSE to run.
```
then you need to extract the AppImage because the system can't do it. Luckily, the AppImage package is self-extractable, so just run:
```bash
./python3.12.AppImage --appimage-extract && mv squashfs-root python3.12
```

From there, to execute a Python script (`.py`), you have to use one of the following methods:

1. `python3.12 script.py` if you have multiple versions of Python installed,
2. `python script.py` if you have only Python 3.12 installed,
3. `./python3.12.AppImage script.py` from within the `VirtualSecretary` folder if you can use the non-extracted AppImage
4. `python3.12/AppRun script.py` from within the `VirtualSecretary` folder if you had to extract the AppImage

In the rest of this document, we will write `PYTHON` as a generic way of calling the interpreter that will need to be replaced by one of the above commands depending on your case.

## Installing the dependencies

The project ships 3 manifests: 
- `requirements.txt` for runtime dependencies,
- `requirements-email.txt` for the email filtering/parsing stack,
- `requirements-dev.txt` for the documentation toolchain.

From the `VirtualSecretary/` folder, run:
```bash
PYTHON -m pip install -r recipe/requirements.txt
PYTHON -m pip install -r recipe/requirements-email.txt
```

If you also want to build and serve the documentation locally:

```bash
pip install -r requirements-dev.txt
mkdocs serve
```

Notable runtime dependencies (already covered by `requirements.txt`):

| Group | Key packages |
|---|---|
| Linear algebra / ML | `numpy`, `scipy`, `numba`, `scikit-learn`, `gensim==4.4` |
| NLP | `nltk`, `fast-langdetect`, `blingfire`, `rank-bm25`, `levenshtein` |
| Network / crawling | `curl_cffi`, `httpx`, `requests`, `protego`, `charset_normalizer` |
| HTML parsing | `beautifulsoup4`, `html5lib` |
| PDF / OCR | `PyMuPDF`, `pdf2image`, `pytesseract`, `pillow==10.4.0`, `opencv-python-headless` |
| Web app | `flask`, `flask-caching`, `markdown` |

!!! warning "Version pins matter"
    `numba==0.61.0` requires `numpy>=2.0,<=2.1`. `pillow==10.4.0` and `pypdf==3.8.1` are pinned to avoid breaking changes in newer releases. Install from `requirements.txt` as-is rather than resolving versions manually.

Tesseract OCR binaries must be installed separately at the OS level for scanned PDF support:

```bash
# Debian / Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra
```

## Notes on servers

!!! note

    This installation procedure was tested on Ubuntu Server 20.04 using an Amazon Elastic Cloud 2 instance, on Fedora Desktop 37 and 38, and on a CPanel-based shared hosting (Debian).


If you use a private or shared hosting, you have access to a server where you may run applications, host websites and mailboxes. It is very desirable to install the Virtual Secretary on the same server hosting your mailbox, as it will be able to filter those email in-place. In that case, for security purposes, it is better to put the application code outside of the HTTP server directories (`var_html`, or `/var-www`, or any directory accessible through a web browser).

CPanel-based shared hostings have limited resources, but my tests show they can still be used for a single user with a couple email addresses. Nowadays, they provide a web terminal allowing to launch command lines, even though they still don't come with `sudo` rights to install software.

If your hosting does not have a web terminal interface, nor SSH access, and is basically just a piece of harddrive connected to the internet with some way to define Cron jobs, you will have to prepare the `VirtualSecretary` folder with the source code and the extracted Python AppImage locally, on your computer, and then to dump it all on the server through FTP.
