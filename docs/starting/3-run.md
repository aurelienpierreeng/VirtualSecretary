# Running the filters

## On user request

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

## With Cron job

You can run your filters every `n` minutes, on Linux desktop and servers, using a Cron job. If you have several devices getting emails, for example, it may be a good idea to run them on your email servers every 10-30 minutes, so all your devices get a clean input.

```bash
contab -e
*/10 * * * * python gazette/virtual-secretary/src/main.py
```
