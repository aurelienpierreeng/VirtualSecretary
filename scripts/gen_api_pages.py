import sys
import importlib
import ast
import pkgutil
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, SRC_ROOT)

BASE_MODULES = ["core", "protocols"]

nav = mkdocs_gen_files.Nav()

def iter_modules(pkg_name: str):
    pkg = importlib.import_module(pkg_name)
    yield pkg_name
    if hasattr(pkg, "__path__"):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            yield m.name

def display_name(base: str, full: str) -> str:
    return full[len(base) + 1:] if full.startswith(base + ".") else full

def get_module_doc(module_name: str) -> str:
    parts = module_name.split(".")
    path = SRC_ROOT.joinpath(*parts).with_suffix(".py")

    try:
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source)
        doc = ast.get_docstring(module)

        if doc:
            return doc.strip().splitlines()[0]

    except Exception:
        pass

    return ""

# -----------------------
# Generate module pages
# -----------------------
for base in BASE_MODULES:
    for mod in iter_modules(base):

        file_path = Path("api") / (mod.replace(".", "/") + ".md")

        with mkdocs_gen_files.open(file_path, "w") as f:
            f.write(f"# {mod}\n\n")
            f.write(f"::: {mod}\n")

        # IMPORTANT: nav path must NOT include "api/" here
        nav[mod] = file_path.as_posix()

# -----------------------
# Generate API index
# -----------------------
index_path = Path("api/index.md")

with mkdocs_gen_files.open(index_path, "w") as f:
    f.write("# API Reference\n\n")

    for base in BASE_MODULES:
        f.write(f"## {base}\n\n")

        for mod in iter_modules(base):
            rel_name = display_name(base, mod)
            doc = get_module_doc(mod)

            page = mod.replace(".", "/") + ".md"

            # IMPORTANT: link is RELATIVE inside api/
            f.write(f"### [{rel_name}]({page})\n")
            if doc:
                f.write(f"{doc}\n")
            f.write("\n")