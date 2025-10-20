from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="deepseek_moe_split",
    ext_modules=[
        CppExtension(
            name="deepseek_moe_split",
            sources=["route_ops.cpp"],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
