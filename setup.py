from sysconfig import get_path
from setuptools import setup, Extension
PATH_PREFIXES = [get_path(p) for p in ["data", "platlib"]]

include_dirs = [
    path
    for prefix in PATH_PREFIXES
    for path in (
        f"{prefix}/include/",
        f"{prefix}/include/tbb",
        f"{prefix}/pybind11/include",
    )
]

modules = []

for code in ["greedy_encoder", "pco_tokenizer"]:
    modules.append(
        Extension(
            name=f"pcatt.{code}",
            sources=[f"pcatt/{code}.cpp"],
            include_dirs=include_dirs,
            library_dirs=[f"{prefix}/lib/" for prefix in PATH_PREFIXES],
            libraries=["tbb"],
            extra_compile_args=["-O3", "-std=c++23"],
            define_macros=[
                ("MAJOR_VERSION", "0"),
                ("MINOR_VERSION", "15"),
            ],
            extra_link_args=[
                "-Wl,-rpath,$ORIGIN/.libs",
                "-Wl,--enable-new-dtags",
            ],
        )
)


setup(ext_modules=modules)
