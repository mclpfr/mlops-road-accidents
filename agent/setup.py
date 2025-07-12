from setuptools import setup, find_packages

setup(
    name="agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'requests',
        'pyyaml',
        'fastapi',
        'uvicorn',
        'websockets',
        'docker',
    ],
)
