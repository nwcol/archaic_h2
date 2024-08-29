from setuptools import setup


setup(
    name="archaic-project",
    version="2024.1",
    author="Nick Collier",
    author_email="nwcollier@wisc.edu",
    packages=[
        "archaic", 
        'archaic.pipeline',
        'archaic.scripts'
    ],
    entry_points={
        "console_scripts": [
            "fit_H2=archaic.scripts.fit_H2:main",
            'plot_H2=archaic.plots.plot_H2:main',
        ]
    }
)
