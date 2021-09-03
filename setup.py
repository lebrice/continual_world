import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
packages = setuptools.find_namespace_packages(include=["continual_world*"])
print("PACKAGES FOUND:", packages)
print(sys.version_info)

import versioneer

setuptools.setup(
    name="continual_world",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="<author name here>",
    author_email="<author email here>",
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
            "sac     = continual_world.sequoia_integration.base_sac_method:SAC",
            "agem    = continual_world.sequoia_integration.agem:AGEM",
            "l2      = continual_world.sequoia_integration.reg_methods:L2Regularization",
            "ewc     = continual_world.sequoia_integration.reg_methods:EWC",
            "mas     = continual_world.sequoia_integration.reg_methods:MAS",
            "packnet = continual_world.sequoia_integration.packnet:PackNet",
            "vcl     = continual_world.sequoia_integration.vcl:VCL",
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        "tensorflow-gpu",
        "simple-parsing",
    ],
)
