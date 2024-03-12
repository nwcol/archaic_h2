from setuptools import setup


setup(
    name="archaic_project",
    version="2024.1",
    author="Nick Collier",
    author_email="nwcollier@wisc.edu",
    packages=["util"],
    scripts=[
        "util/scripts/simplify_vcf.py"
    ]
)
