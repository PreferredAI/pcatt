from sysconfig import get_path
from setuptools import setup, Extension
from pathlib import Path

PATH_PREFIXES = [get_path(p) for p in ['data', 'platlib']]

modules = []
include_dirs = [
    path
    for prefix in PATH_PREFIXES
    for path in [
        f"{prefix}/include/",
        f"{prefix}/include/tbb",
        f"{prefix}/pybind11/include"
    ]
]

for code in ["greedy_builder", "greedy_encoder", "pco_tokenizer"]:
    modules.append(
        Extension(
            f"pcatt.{code}",
            extra_compile_args=["-O3", "-std=c++23"],
            define_macros=[("MAJOR_VERSION", "0"), ("MINOR_VERSION", "14-beta")],
            include_dirs=include_dirs,
            library_dirs=[f"{prefix}/lib/" for prefix in PATH_PREFIXES] ,
            libraries=["tbb"],
            sources=[f"pcatt/{code}.cpp"],
        )
    )

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="greedtok",
    version="0.14-beta",
    description="Partition Cover Approach to Tokenization",
    author="JP Lim",
    author_email="jiapeng.lim.2021@phdcs.smu.edu.sg",
    license="MIT",
    setup_requires=["pybind11", "tbb-devel", "transformers>=4.4"],
    url="https://github.com/PreferredAI/pcatt/",
    download_url="https://github.com/PreferredAI/pcatt/archive/refs/tags/v0.14-beta5.tar.gz",
    ext_modules=modules,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
