try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name="IOModel",
    packages=["IOModel", "IOModel.models", "IOModel.matrix_balancing"],
    version="0.1",
    description="input-output model",
    long_description="input-output model, complete with matrix balancing",
    author="me",
    author_email="tony.barnett@cloudbuy.com",
    url="localhost",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3"
    ], requires=['numpy']
)
