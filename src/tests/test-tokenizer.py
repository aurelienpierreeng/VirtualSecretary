"""
Test the sentences and word tokenization.

"""

import os
import sys

import nltk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import nlp
from core import utils

text = """I’m trying to get darktable working with opencl. Since all opencl options in preferences are greyed out with the text “not available”, I’m led to believe there “something outside of darktable” that I need to take care of.

It seems I have two darktable binaries on the system, not sure why.
The one I’ve been using so far, that works is:
/var/lib/flatpak/exports/bin/org.darktable.Darktable

This one fails with “libOpenEXR-3_1.so.30: cannot open shared object file: No such file or directory”:
/var/lib/flatpak/app/org.darktable.Darktable/current/e4c1a74e092a04300
c463ed8d03dbeb4de4da8dada632470533d3e41c5b59d17/files/bin/darktable

Well I’m not even sure I’m supposed to run that second binary, but at least it gives me opencl-hopes, especially when I see this:

atte@ajstrup:~$ ll /var/lib/flatpak/app/org.darktable.Darktable/current/e4c1a74e092a04300c463ed8d03dbeb4de4da8dada632470533d3e41c5b59d17/files/bin/
total 6.7M
-rwxr-xr-x 2 root root 23K Jan 1 1970 cd-convert
-rwxr-xr-x 2 root root 27K Jan 1 1970 cd-create-profile
-rwxr-xr-x 2 root root 31K Jan 1 1970 cd-fix-profile
-rwxr-xr-x 2 root root 15K Jan 1 1970 cd-iccdump
-rwxr-xr-x 2 root root 27K Jan 1 1970 cd-it8
-rwxr-xr-x 2 root root 67K Jan 1 1970 colormgr
-rwxr-xr-x 2 root root 15K Jan 1 1970 darktable
-rwxr-xr-x 2 root root 99K Jan 1 1970 darktable-chart
-rwxr-xr-x 2 root root 31K Jan 1 1970 darktable-cli
-rwxr-xr-x 2 root root 15K Jan 1 1970 darktable-cltest
-rwxr-xr-x 2 root root 23K Jan 1 1970 darktable-cmstest
-rwxr-xr-x 2 root root 23K Jan 1 1970 darktable-generate-cache
-rwxr-xr-x 2 root root 895K Jan 1 1970 darktable-rs-identify
-rwxr-xr-x 2 root root 23K Jan 1 1970 dec265
-rwxr-xr-x 2 root root 31K Jan 1 1970 enc265
-rwxr-xr-x 2 root root 3.4M Jan 1 1970 git
-rwxr-xr-x 2 root root 160K Jan 1 1970 git-cvsserver
lrwxrwxrwx 1 root root 3 Feb 20 11:27 git-receive-pack → git
-rwxr-xr-x 2 root root 1.9M Jan 1 1970 git-shell
lrwxrwxrwx 1 root root 3 Feb 20 11:27 git-upload-archive → git
lrwxrwxrwx 1 root root 3 Feb 20 11:27 git-upload-pack → git
-rwxr-xr-x 2 root root 15K Jan 1 1970 gm
-rwxr-xr-x 2 root root 23K Jan 1 1970 gusbcmd
-rwxr-xr-x 2 root root 19K Jan 1 1970 hdrcopy

ll /var/lib/flatpak/app/org.darktable.Darktable/current/e4c1a74e092a04300c463ed8d03dbeb4de4da8dada632470533d3e41c5b59d17/files/lib
total 38M
drwxr-xr-x 4 root root 4.0K Jan 1 1970 darktable
drwxr-xr-x 2 root root 4.0K Jan 1 1970 debug
drwxr-xr-x 3 root root 4.0K Jan 1 1970 GraphicsMagick-1.3.38
lrwxrwxrwx 1 root root 17 Feb 20 11:27 libavif.so.11 → libavif.so.11.0.0
-rwxr-xr-x 2 root root 7.8M Jan 1 1970 libavif.so.11.0.0
lrwxrwxrwx 1 root root 18 Feb 20 11:27 libcolord-gtk.so → libcolord-gtk.so.1
lrwxrwxrwx 1 root root 22 Feb 20 11:27 libcolord-gtk.so.1 → libcolord-gtk.so.1.0.3
-rwxr-xr-x 2 root root 35K Jan 1 1970 libcolord-gtk.so.1.0.3
lrwxrwxrwx 1 root root 21 Feb 20 11:27 libcolordprivate.so → libcolordprivate.so.2
lrwxrwxrwx 1 root root 25 Feb 20 11:27 libcolordprivate.so.2 → libcolordprivate.so.2.0.5
-rwxr-xr-x 2 root root 236K Jan 1 1970 libcolordprivate.so.2.0.5
lrwxrwxrwx 1 root root 14 Feb 20 11:27 libcolord.so → libcolord.so.2
lrwxrwxrwx 1 root root 18 Feb 20 11:27 libcolord.so.2 → libcolord.so.2.0.5
-rwxr-xr-x 2 root root 368K Jan 1 1970 libcolord.so.2.0.5
lrwxrwxrwx 1 root root 16 Feb 20 11:27 libcolorhug.so → libcolorhug.so.2
lrwxrwxrwx 1 root root 20 Feb 20 11:27 libcolorhug.so.2 → libcolorhug.so.2.0.5
-rwxr-xr-x 2 root root 95K Jan 1 1970 libcolorhug.so.2.0.5
lrwxrwxrwx 1 root root 14 Feb 20 11:27 libexiv2.so → libexiv2.so.27
-rwxr-xr-x 2 root root 3.2M Jan 1 1970 libexiv2.so.0.27.5
lrwxrwxrwx 1 root root 18 Feb 20 11:27 libexiv2.so.27 → libexiv2.so.0.27.5
-rwxr-xr-x 2 root root 12M Jan 1 1970 libgmic.so.1
drwxr-xr-x 3 root root 4.0K Jan 1 1970 libgphoto2
drwxr-xr-x 3 root root 4.0K Jan 1 1970 libgphoto2_port
lrwxrwxrwx 1 root root 25 Feb 20 11:27 libgphoto2_port.so → libgphoto2_port.so.12.1.0
lrwxrwxrwx 1 root root 25 Feb 20 11:27 libgphoto2_port.so.12 → libgphoto2_port.so.12.1.0
-rwxr-xr-x 2 root root 51K Jan 1 1970 libgphoto2_port.so.12.1.0
lrwxrwxrwx 1 root root 19 Feb 20 11:27 libgphoto2.so → libgphoto2.so.6.3.0
lrwxrwxrwx 1 root root 19 Feb 20 11:27 libgphoto2.so.6 → libgphoto2.so.6.3.0
-rwxr-xr-x 2 root root 159K Jan 1 1970 libgphoto2.so.6.3.0
lrwxrwxrwx 1 root root 29 Feb 20 11:27 libGraphicsMagick++.so → libGraphicsMagick++.so.12.6.0
lrwxrwxrwx 1 root root 27 Feb 20 11:27 libGraphicsMagick.so → libGraphicsMagick.so.3.24.0
lrwxrwxrwx 1 root root 29 Feb 20 11:27 libGraphicsMagick++.so.12 → libGraphicsMagick++.so.12.6.0
-rwxr-xr-x 2 root root 519K Jan 1 1970 libGraphicsMagick++.so.12.6.0
lrwxrwxrwx 1 root root 27 Feb 20 11:27 libGraphicsMagick.so.3 → libGraphicsMagick.so.3.24.0
-rwxr-xr-x 2 root root 2.9M Jan 1 1970 libGraphicsMagick.so.3.24.0
lrwxrwxrwx 1 root root 30 Feb 20 11:27 libGraphicsMagickWand.so → libGraphicsMagickWand.so.2.9.7
lrwxrwxrwx 1 root root 30 Feb 20 11:27 libGraphicsMagickWand.so.2 → libGraphicsMagickWand.so.2.9.7
-rwxr-xr-x 2 root root 199K Jan 1 1970 libGraphicsMagickWand.so.2.9.7
lrwxrwxrwx 1 root root 12 Feb 20 11:27 libgusb.so → libgusb.so.2
lrwxrwxrwx 1 root root 17 Feb 20 11:27 libgusb.so.2 → libgusb.so.2.0.10
-rwxr-xr-x 2 root root 75K Jan 1 1970 libgusb.so.2.0.10
lrwxrwxrwx 1 root root 12 Feb 20 11:27 libheif.so → libheif.so.1
lrwxrwxrwx 1 root root 19 Feb 20 11:27 libheif.so.1 → libheif.so.1.12.0.0
-rwxr-xr-x 2 root root 527K Jan 1 1970 libheif.so.1.12.0.0
lrwxrwxrwx 1 root root 16 Feb 20 11:27 libIex-3_1.so → libIex-3_1.so.30
lrwxrwxrwx 1 root root 20 Feb 20 11:27 libIex-3_1.so.30 → libIex-3_1.so.30.3.0
-rwxr-xr-x 2 root root 563K Jan 1 1970 libIex-3_1.so.30.3.0
lrwxrwxrwx 1 root root 13 Feb 20 11:27 libIex.so → libIex-3_1.so
lrwxrwxrwx 1 root root 22 Feb 20 11:27 libIlmThread-3_1.so → libIlmThread-3_1.so.30
lrwxrwxrwx 1 root root 26 Feb 20 11:27 libIlmThread-3_1.so.30 → libIlmThread-3_1.so.30.3.0
-rwxr-xr-x 2 root root 35K Jan 1 1970 libIlmThread-3_1.so.30.3.0
lrwxrwxrwx 1 root root 19 Feb 20 11:27 libIlmThread.so → libIlmThread-3_1.so
lrwxrwxrwx 1 root root 18 Feb 20 11:27 libImath-3_1.so → libImath-3_1.so.29
lrwxrwxrwx 1 root root 22 Feb 20 11:27 libImath-3_1.so.29 → libImath-3_1.so.29.2.0
-rwxr-xr-x 2 root root 327K Jan 1 1970 libImath-3_1.so.29.2.0
lrwxrwxrwx 1 root root 15 Feb 20 11:27 libImath.so → libImath-3_1.so
lrwxrwxrwx 1 root root 14 Feb 20 11:27 libjasper.so → libjasper.so.4
lrwxrwxrwx 1 root root 18 Feb 20 11:27 libjasper.so.4 → libjasper.so.4.0.0
-rwxr-xr-x 2 root root 311K Jan 1 1970 libjasper.so.4.0.0
lrwxrwxrwx 1 root root 13 Feb 20 11:27 libjxl.so → libjxl.so.0.7
lrwxrwxrwx 1 root root 15 Feb 20 11:27 libjxl.so.0.7 → libjxl.so.0.7.0
-rwxr-xr-x 2 root root 3.6M Jan 1 1970 libjxl.so.0.7.0
lrwxrwxrwx 1 root root 21 Feb 20 11:27 libjxl_threads.so → libjxl_threads.so.0.7
lrwxrwxrwx 1 root root 23 Feb 20 11:27 libjxl_threads.so.0.7 → libjxl_threads.so.0.7.0
-rwxr-xr-x 2 root root 23K Jan 1 1970 libjxl_threads.so.0.7.0
lrwxrwxrwx 1 root root 15 Feb 20 11:27 liblensfun.so → liblensfun.so.1
-rwxr-xr-x 2 root root 131K Jan 1 1970 liblensfun.so.0.3.3
lrwxrwxrwx 1 root root 19 Feb 20 11:27 liblensfun.so.1 → liblensfun.so.0.3.3
-rwxr-xr-x 2 root root 811K Jan 1 1970 liblibde265.so
lrwxrwxrwx 1 root root 15 Feb 20 11:27 liblua.so → liblua.so.5.4.4
lrwxrwxrwx 1 root root 15 Feb 20 11:27 liblua.so.5.4 → liblua.so.5.4.4
-rw-r–r-- 2 root root 276K Jan 1 1970 liblua.so.5.4.4
lrwxrwxrwx 1 root root 20 Feb 20 11:27 libOpenEXR-3_1.so → libOpenEXR-3_1.so.30
lrwxrwxrwx 1 root root 24 Feb 20 11:27 libOpenEXR-3_1.so.30 → libOpenEXR-3_1.so.30.3.0
-rwxr-xr-x 2 root root 3.3M Jan 1 1970 libOpenEXR-3_1.so.30.3.0
lrwxrwxrwx 1 root root 24 Feb 20 11:27 libOpenEXRCore-3_1.so → libOpenEXRCore-3_1.so.30
lrwxrwxrwx 1 root root 28 Feb 20 11:27 libOpenEXRCore-3_1.so.30 → libOpenEXRCore-3_1.so.30.3.0
-rwxr-xr-x 2 root root 515K Jan 1 1970 libOpenEXRCore-3_1.so.30.3.0
lrwxrwxrwx 1 root root 21 Feb 20 11:27 libOpenEXRCore.so → libOpenEXRCore-3_1.so
lrwxrwxrwx 1 root root 17 Feb 20 11:27 libOpenEXR.so → libOpenEXR-3_1.so
lrwxrwxrwx 1 root root 24 Feb 20 11:27 libOpenEXRUtil-3_1.so → libOpenEXRUtil-3_1.so.30
lrwxrwxrwx 1 root root 28 Feb 20 11:27 libOpenEXRUtil-3_1.so.30 → libOpenEXRUtil-3_1.so.30.3.0
-rwxr-xr-x 2 root root 227K Jan 1 1970 libOpenEXRUtil-3_1.so.30.3.0
lrwxrwxrwx 1 root root 21 Feb 20 11:27 libOpenEXRUtil.so → libOpenEXRUtil-3_1.so
lrwxrwxrwx 1 root root 25 Feb 20 11:27 libosmgpsmap-1.0.so → libosmgpsmap-1.0.so.1.1.0
lrwxrwxrwx 1 root root 25 Feb 20 11:27 libosmgpsmap-1.0.so.1 → libosmgpsmap-1.0.so.1.1.0
-rwxr-xr-x 2 root root 111K Jan 1 1970 libosmgpsmap-1.0.so.1.1.0
lrwxrwxrwx 1 root root 16 Feb 20 11:27 libportmidi.so → libportmidi.so.2
lrwxrwxrwx 1 root root 20 Feb 20 11:27 libportmidi.so.2 → libportmidi.so.2.0.3
-rwxr-xr-x 2 root root 51K Jan 1 1970 libportmidi.so.2.0.3
lrwxrwxrwx 1 root root 19 Feb 20 11:27 libusb-1.0.so → libusb-1.0.so.0.3.0
lrwxrwxrwx 1 root root 19 Feb 20 11:27 libusb-1.0.so.0 → libusb-1.0.so.0.3.0
-rwxr-xr-x 2 root root 123K Jan 1 1970 libusb-1.0.so.0.3.0
drwxr-xr-x 3 root root 4.0K Jan 1 1970 lua
drwxr-xr-x 2 root root 4.0K Jan 1 1970 udev
drwxr-xr-x 3 root root 4.0K Jan 1 1970 x86_64-linux-gnu

So it seems I do have opencl available, but not really. Any clues on how to proceed, any libraries or opencl stuff that needs to be installed via apt-get on the system?

Cool, thank, I’ll do that!

Anyways, I build darktable from source and OpenCL is still not available, any idea if it’s possible:

atte@ajstrup:~/software/darktable$ /opt/darktable/bin/darktable-cltest
[dt_get_sysresource_level] switched to 1 as `default’
total mem: 7848MB
mipmap cache: 981MB
available mem: 3924MB
singlebuff: 61MB
OpenCL tune mem: OFF
OpenCL pinned: OFF
[opencl_init] opencl related configuration options:
[opencl_init] opencl: ON
[opencl_init] opencl_scheduling_profile: ‘default’
[opencl_init] opencl_library: ‘default path’
[opencl_init] opencl_device_priority: ‘/!0,///!0,*’
[opencl_init] opencl_mandatory_timeout: 400
[opencl_init] opencl library ‘libOpenCL.so.1’ found on your system and loaded
[opencl_init] found 1 platform
[opencl_init] no devices found for Mesa (vendor) - Clover (name)
[opencl_init] found 0 device
[opencl_init] FINALLY: opencl is NOT AVAILABLE on this system.
[opencl_init] initial status of opencl enabled flag is OFF.

atte@ajstrup:~/software/darktable$ mlocate libOpenCL.so.1
/usr/lib/i386-linux-gnu/libOpenCL.so.1
/usr/lib/i386-linux-gnu/libOpenCL.so.1.0.0
/usr/lib/x86_64-linux-gnu/libOpenCL.so.1
/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0
/var/lib/flatpak/runtime/org.gnome.Platform/x86_64/40/83cb0a34f6f1671c54e72706e829a8070a9a3a4add2c812e2b50fbfe6e368df2/files/lib/x86_64-linux-gnu/libOpenCL.so.1
/var/lib/flatpak/runtime/org.gnome.Platform/x86_64/40/83cb0a34f6f1671c54e72706e829a8070a9a3a4add2c812e2b50fbfe6e368df2/files/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0
/var/lib/flatpak/runtime/org.gnome.Platform/x86_64/43/01e39e8c8bfdbf6effded1ca6c5a88b4177b920d4761dc20d8a334d675d051a5/files/lib/x86_64-linux-gnu/libOpenCL.so.1
/var/lib/flatpak/runtime/org.gnome.Platform/x86_64/43/01e39e8c8bfdbf6effded1ca6c5a88b4177b920d4761dc20d8a334d675d051a5/files/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0
/var/lib/flatpak/runtime/org.kde.Platform/x86_64/6.3/da5db79df3afb70fe080b06f44b84ef1db3400ea8eb8cba77c22ab07c9065f64/files/lib/x86_64-linux-gnu/libOpenCL.so.1
/var/lib/flatpak/runtime/org.kde.Platform/x86_64/6.3/da5db79df3afb70fe080b06f44b84ef1db3400ea8eb8cba77c22ab07c9065f64/files/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0

I don't seem to be able to download the darktable.zip archive either.

I need to borrow $12 or 12 €.

Ask @aurelien.
\n\n

28 Feb 20 11:27 libOpenEXRCore-3_1.so.30.
\n\n
515K Jan 1 1970 libOpenEXRCore-3_1.so.30.3.0.
\n\n
21 Feb 20 11:27.
\n\n
"""

clean_text = utils.typography_undo(text)
language = nlp.guess_language(clean_text)
clean_text = nlp.prefilter_tokenizer(clean_text)
sentences = [nlp.tokenize_sentences(sentence, language) for sentence in nlp.split_sentences(clean_text, language)]

for sentence in sentences:
  print(sentence)
