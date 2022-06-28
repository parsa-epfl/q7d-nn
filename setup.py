import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="q7d-nn",
    version="0.0.1",
    author="PARSA EPFL",
    author_email="canberk.sonmez@epfl.ch",
    description="Quantized NN (Q7D-NN) is our awesome framework for testing quantization in single- and multi-node DNN training settings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parsa-epfl/q7d-nn",
    project_urls={
        "Bug Tracker": "https://github.com/parsa-epfl/q7d-nn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.8",
    package_data={  },
    install_requires=[
        "torch",
    ],
    license="MIT"
)
