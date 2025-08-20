from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

setup(
    name="deepseek_moe",
    ext_modules=[
        CppExtension(
            name="deepseek_moe",
            sources=[os.path.join(os.path.dirname(__file__), "moe.cpp")],
            include_dirs=[os.path.join(ROOT, "common")],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
