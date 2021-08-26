import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
packages = setuptools.find_namespace_packages(include=["continual_world*"])
print("PACKAGES FOUND:", packages)
print(sys.version_info)

setuptools.setup(
    name="continual_world",
    version="0.0.1",
    author="",
    author_email="<TODO>",
    description="Continual World: A Robotic Benchmark For Continual Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awarelab/continual_world",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "Method": [
            # TODO: Add the methods to the package!
            # "cndpm = cn_dpm.cndpm_method:CNDPM",
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        "tensorflow-gpu",
        "simple-parsing",
    ],
)
