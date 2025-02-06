import setuptools
from Cython.Build import cythonize

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="msadaptor",
    version="0.0.1",
    author="msadaptor",
    author_email="lvyufeng@cqu.edu.cn",
    description="msadaptor project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lvyufeng/msadaptor",
    project_urls={
        "Bug Tracker": "https://github.com/lvyufeng/msadaptor/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["mindtorch", "torch", "torch_npu", 'torchvision'],
    package_dir={"": "mindtorch"},
    package_data={'': ['*', '*/*', '*/*/*','*/*/*/*','*/*/*/*/*','*/*/*/*/*/*']},
    python_requires=">=3.9",
    install_requires=[
        "mindspore>=2.4",
        "requests",
    ],
    ext_modules=cythonize("mindtorch/torch/dispatcher.pyx")
)
