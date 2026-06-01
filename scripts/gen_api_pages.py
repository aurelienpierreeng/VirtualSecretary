import sys
import ast
import pkgutil
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, SRC_ROOT)

BASE_MODULES = ["core", "protocols"]

nav = mkdocs_gen_files.Nav()

def iter_modules(base: str):
    base_path = SRC_ROOT / base

    for module in pkgutil.walk_packages(
        [str(base_path)],
        prefix=f"{base}."
    ):
        yield module.name


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

        rel_name = display_name(base, mod)
        nav[base, rel_name] = Path(mod.replace(".", "/")).with_suffix(".md").as_posix()

# -----------------------
# Generate API nav
# -----------------------
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.write("* [API Reference](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())

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
