from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "mcts_cpp",
        ["cpp/mcts_module.cpp"],
        include_dirs=["cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="alphazero-chess",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=[],
)
