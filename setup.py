from setuptools import setup, find_packages

requirements = [
    'numpy', 'pandas', 'pyyaml', 'trimesh', 'wandb', 'torch', 'pytorch_lightning',
    'tqdm', 'yacs', 'pyntcloud', 'pytorch3d', 'einops'
]


def get_version():
    version_file = 'multi_part_assembly/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name="multi_part_assembly",
    version=get_version(),
    description="Code for Learning 3D Geometric Shape Assembly",
    long_description="Code for Learning 3D Geometric Shape Assembly",
    author="Ziyi Wu",
    author_email="ziyiwu@cs.toronto.edu",
    license="",
    url="",
    keywords="multi part shape assembly",
    packages=find_packages(),
    install_requires=requirements,
)
