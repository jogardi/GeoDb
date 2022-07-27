from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="geodbcpp",
    ext_modules=[cpp_extension.CppExtension("geodbcpp", ["geodbcpp.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
