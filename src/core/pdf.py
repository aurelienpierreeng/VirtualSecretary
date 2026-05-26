"""PDF parsing utils, including OCR.

© 2024 - Aurélien Pierre
"""

from io import BytesIO
import pytesseract
import cv2
import pdf2image
import requests

import pymupdf
pymupdf.TOOLS.set_icc(False)

import fitz

from PIL import Image
import numpy as np
import regex as re
from datetime import datetime

from .types import web_page
from .network import get_url, DelayedClass
from .patterns import HYPHENIZED
from .utils import clean_whitespaces

def ocr_pdf(document: bytes, 
            output_images: bool = False, 
            path: str = None,
            repair: int = 1, 
            upscale: int = 3, 
            contrast: float = 1.5, 
            sharpening: float = 1.2, 
            threshold: float = 0.4,
            tesseract_lang: str = "eng+fra+equ", 
            tesseract_bin: str = None) -> str:
    """Extract text from PDF using OCR through [Tesseract](https://github.com/tesseract-ocr/tesseract). Both the binding [Python package PyTesseract](https://pypi.org/project/pytesseract/#installation) __and__ the [Tesseract binaries](https://tesseract-ocr.github.io/tessdoc/Installation.html) need to be installed.

    To run on a server where you don't have `sudo` access to install package, you will need to download the [AppImage package](https://tesseract-ocr.github.io/tessdoc/Installation.html#appimage) and pass its path to the `tesseract_bin` argument.

    Tesseract uses machine-learning to identify words and needs the relevant language models to be installed on the system as well. Linux packaged version of Tesseract seem to generally ship French, English and equations (math) models by default. Other languages need to be installed manually,  see [Tesseract docs](https://tesseract-ocr.github.io/tessdoc/Data-Files#data-files-for-version-400-november-29-2016) for available packages. Use `pytesseract.get_languages(config='')` to list available language packages installed locally.

    The OCR is preceeded by an image processing step aiming at text reconstruction, by sharpening, increasing contrast and iteratively reconstructing holes in letters using an inpainting method in wavelets space. This is computationaly expensive, which may not be suitable to run on server.

    Arguments:
        document: the PDF document to open.
        output_images: set to `True`, each page of the document is saved as PNG in the `path` directory before and after contrast enhancement. This is useful to tune the image contrast and sharpness enhancements, prior to OCR.
        repair: number of iterations of enhancements (sharpening, contrast and inpainting) to perform. More iterations take longer, too many iterations might simplify their geometry (as if they were fluid and would drip, removing corners and pointy ends) in a way that actually degrades OCR.
        upscale: upscaling factor to apply before enhancement. This can help recovering ink leaks but takes more memory and time to compute.
        contrast: `1.0` is the neutral value. Moves RGB values farther away from the threshold.
        sharpening: `1.0` is the neutral value. Increases sharpness. Values too high can produce ringing (replicated ghost edges).
        threshold: the reference value (fulcrum) for contrast enhancement. Good values are typically in the range 0.20-0.50.
        tesseract_lang: the Tesseract command argument `-l` defining languages models to use for OCR. Languages are referenced by their 3-letters ISO-something code. See [Tesseract doc](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html#using-multiple-languages) for syntax and meaning. You can mix several languages by joining them with `+`.
        tesseract_bin: the path to the Tesseract executable if it is not in the global CLI path. This is passed as-is to `pytesseract.pytesseract.tesseract_cmd` of the [PyTesseract](https://pypi.org/project/pytesseract/) binding library.

    Returns:
        All the retrieved text from all the PDF pages as a single string. No pagination is done.

    Raises:
        RuntimeError: when using a language package is attempted while Tesseract has no such package installed.
    """
    count = 0
    content = ""

    if tesseract_bin:
        pytesseract.pytesseract.tesseract_cmd = tesseract_bin

    tesseract_langs = tesseract_lang.split("+")
    for _lang in tesseract_langs:
        if _lang not in pytesseract.get_languages(config=''):
            raise RuntimeError("The Tesseract language package `%s` is not installed on this system. Visit https://tesseract-ocr.github.io/tessdoc/Data-Files" % _lang)

    for image in pdf2image.convert_from_bytes(document):
        if output_images and path:
            image.save(path + "-" + str(count) + "-in.png")

        # Convert image to grayscale if it's RGB(a)
        img = np.array(image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Convert to float, un-gamma and invert
        gray = (gray / 255.)**2.4

        # Upsample
        gray = cv2.resize(gray, (gray.shape[1] * upscale,
                                 gray.shape[0] * upscale))

        # Iterative laplacian pyramid sharpening + 4th order diffusing
        for iter in range(repair):
            LF = gray
            residual = np.zeros_like(gray)
            for i in [3, 5, 9]:
                LF_2 = cv2.GaussianBlur(LF, (i, i), 0)
                HF = LF - LF_2
                residual += (cv2.GaussianBlur(HF, (i, i), 0) + HF) / 2 * sharpening
                LF = LF_2

            # Reconstruct the pyramid
            gray = np.clip(residual + LF, 0, np.inf)

            # Contraste : engraisse le noir
            gray = (gray / threshold)**contrast * threshold

        # Convert back to uint8 and redo gamma
        gray = (np.clip(gray, 0, 1)**(1./2.4) * 255).astype(np.uint8)

        if output_images and path:
            to_save = Image.fromarray(cv2.resize(gray, (img.shape[1], img.shape[0])))
            to_save.save(path + "-" + str(count) + "-out.png")
            count += 1

        # OCR
        page = pytesseract.image_to_string(gray, lang=tesseract_lang).strip("\n ")
        content += "\n" + page
        #print(page)

    return content.strip("\n ")


def _get_pdf_outline(doc: fitz.Document, document_title: str) -> tuple[list[str], list[int]]:
    """Return a list of chapter titles and 0-based page bounds using PyMuPDF's TOC.

    The TOC returned by `doc.get_toc()` is a list of [level, title, page] entries
    where `page` is 1-based. We convert pages to 0-based indices and prefix each
    title with the document title for uniqueness (matching previous behaviour).
    """
    toc = doc.get_toc()
    chapters_titles: list[str] = []
    chapters_bounds: list[int] = []

    for level, title, page in toc:
        chapters_titles.append(f"{document_title} | {title}")
        # convert 1-based page number to 0-based index
        chapters_bounds.append(max(0, page - 1))

    return [document_title] + chapters_titles, [0] + chapters_bounds


def _extract_text_from_page(page: fitz.Page, min_chars: int = 20) -> str:
    """Robustly extract text from a PyMuPDF `page`.

    Strategy:
    - Try `page.get_text("text")` (fast)
    - If too short, try `page.get_text("dict")` and collect spans
    - Then try `page.get_text("words")` and join words in reading order
    """
    text = page.get_text("text") or ""
    text = text.strip()

    if len(text) >= min_chars:
        return text

    # Try dict/blocks -> spans
    d = page.get_text("dict")
    parts: list[str] = []
    for b in d.get("blocks", []):
        if b.get("type") == 0:
            for line in b.get("lines", []):
                span_text = "".join([s.get("text", "") for s in line.get("spans", [])])
                if span_text.strip():
                    parts.append(span_text)
    text2 = "\n".join(parts).strip()
    if len(text2) >= min_chars:
        return text2

    # Try words order
    words = page.get_text("words")
    if words:
        words_sorted = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
        text3 = " ".join(w[4] for w in words_sorted).strip()
        if len(text3) >= min_chars:
            return text3



def get_pdf_content(url: str,
                    lang: str,
                    delay: DelayedClass,
                    file_path: str = None,
                    process_outline: bool = True,
                    category: str = None,
                    ocr: int = 1,
                    max_size: int = 20,
                    max_pages: int = 20,
                    custom_header: dict = {},
                    **kwargs) -> list[web_page]:
    """Retrieve a PDF document through the network with HTTP GET or from the local filesystem, and parse its text content, using OCR if needed. This needs a functionnal network connection if `file_path` is not provided.

    Arguments:
        url: the online address of the document, or the downloading page if the doc is not directly accessible from a GET request (for some old-schools website where downloads are inited from a POST request to some PHP form handler, or publications behind a paywall).
        lang: the ISO code of the language.
        file_path: local path to the PDF file if the URL can't be directly fetched by GET request. The content will be extracted from the local file but the original/remote URL will still be referenced as the source.
        process_outline: set to `True` to split the document according to its outline (table of content), so each section will be in fact a document in itself. PDF pages are processed in full, so sections are at least equal to a page length and there will be some overlapping.
        category: arbitrary category or label set by user
        ocr:
            - `0` disables any attempt at using OCR,
            - `1` enables OCR through Tesseract if no text was found in the PDF document
            - `2` forces OCR through Tesseract even when text was found in the PDF document.
        See [core.crawler.ocr_pdf][] for info regarding the Tesseract environment. You will need to manually disable
        max_size: when attempting OCR on PDF files, files larger than this value (in MiB) will be ignored.
        max_pages: when attempting OCR on PDF files, files having more pages than this value will be ignored.
        custom_header: option HTTP headers to form the request that will download the PDF

    Other parameters:
        **kwargs: directly passed-through to [core.crawler.ocr_pdf][]. See this function documentation for more info.

    Returns:
        a list of [core.crawler.web_page][] objects holding the text content and the PDF metadata
    """
    try:
        # Open the document from local or remote storage
        document : BytesIO
        if not file_path:
            content, url, status, encoding, apparent_encoding = get_url(url, delay, timeout=60, custom_header=custom_header)

            if status != 200:
                print("couldn't download %s" % url)
                return []

            document = BytesIO(content)
        else:
            document = open(file_path, "rb")

    except Exception as e:
        print("PDF handling error:", e)
        return []

    max_pdf_size = max_size * 1024 * 1024

    # Get the size of the blob
    pos = document.tell()
    document.seek(0, 2)
    pdf_size = document.tell()
    document.seek(pos)

    blob = document.read() # need to backup PDF content here because reader will consume it

    try:
        doc = fitz.open(stream=blob, filetype="pdf")
    except Exception as e:
        print(e)
        return []

    if not doc:
        return []

    # Metadata access
    try:
        meta = doc.metadata or {}
    except:
        meta = {}

    try:
        date = meta.get("creationDate") or meta.get("CreationDate")
    except:
        date = None

    if isinstance(date, datetime):
        date = date.isoformat()

    title = meta.get("title") if meta.get("title") else url.split("/")[-1]
    excerpt = meta.get("subject")

    # Check if the PDF contains text. Use a robust per-page extractor that
    # falls back to blocks/words then per-page OCR when necessary.
    try:
        content = clean_whitespaces("\n".join([_extract_text_from_page(page) for page in doc]).strip("\n "))
    except:
        try:
            doc.close()
        except:
            pass
        return []

    if ((ocr == 1 and len(content) < 20) or ocr == 2):
        if pdf_size <= max_pdf_size and doc.page_count <= max_pages:
            # No text, retry with OCR
            try:
                content = clean_whitespaces(ocr_pdf(blob, path=file_path, **kwargs))
            except Exception as e:
                print(e)
        else:
            filename = file_path if file_path else url
            print(f"PDF file {filename} is too big to be OCR-ed ({doc.page_count} pages, "
                f"{pdf_size / (1024 * 1024)} MiB, {len(content)} characters found). Change your settings if you need it.")

    # Ugly code ahead.
    # FIXME: make that mess more rigorous.
    try:
        # Need to protect TOC retrieval: the web is full of shitty PDFs.
        if doc.get_toc() and process_outline:
            # Save each outline section in a different document
            results = []
            chapters_titles, chapters_bounds = _get_pdf_outline(doc, title)

            if len(chapters_bounds) == 0 and len(content) > 0:
                result = web_page(title=title,
                                    url=url,
                                    date=date,
                                    content=content,
                                    excerpt=excerpt,
                                    h1={},
                                    h2={},
                                    lang=lang,
                                    category=category)
                print("found 1 PDF")
                doc.close()
                return [result]

            for i in range(0, len(chapters_bounds) - 1):
                n_start = chapters_bounds[i]
                n_end = min(chapters_bounds[i + 1], doc.page_count)
                parts = []
                for p in range(n_start, n_end):
                    parts.append(_extract_text_from_page(doc.load_page(p)))
                chapter_content = clean_whitespaces("\n".join(parts).strip("\n "))
                chapter_content = HYPHENIZED.sub("", chapter_content, concurrent=True)

                if chapter_content:
                    # Make up a page anchor to make URLs to document sections unique
                    # since that's what is used as key for dictionaries. Also, Chrome and Acrobat
                    # will be able to open PDF files at the right page with this anchor.
                    result = web_page(title=chapters_titles[i],
                                        url=f"{url}#page={i + 1}",
                                        date=date,
                                        content=chapter_content,
                                        excerpt=None,
                                        h1={},
                                        h2={},
                                        lang=lang,
                                        category=category)
                    results.append(result)

            print("found", i, "PDF chapters")
            doc.close()
            return results

        else:
            # Whether or not text comes from OCR, if we save it in one chunk, do it now and exit.
            content = clean_whitespaces(HYPHENIZED.sub("", content))

            if content:
                result = web_page(title=title,
                                    url=url,
                                    date=date,
                                    content=content,
                                    excerpt=excerpt,
                                    h1={},
                                    h2={},
                                    lang=lang,
                                    category=category)
                print("found 1 PDF")
                doc.close()
                return [result]
    except:
        # Whether or not text comes from OCR, if we save it in one chunk, do it now and exit.
        content = clean_whitespaces(HYPHENIZED.sub("", content))

        if content:
            result = web_page(title=title,
                                url=url,
                                date=date,
                                content=content,
                                excerpt=excerpt,
                                h1={},
                                h2={},
                                lang=lang,
                                category=category)
            print("found 1 PDF")
            doc.close()
            return [result]


    doc.close()
    return []
