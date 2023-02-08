from setuptools import setup

name = 'plasticity'
here = os.path.abspath(os.path.dirname(__file__))
version_info = {}
with open(os.path.join(here, name, 'version_info.py')) as fp:
    exec(fp.read(), version_info)
version = version_info['_version']

setup(
        name=name,
        version=str(version),
        description='Meta-Learning Plasticity',
        url='https://github.com/yashsmehta/plasticity',
        author='Yash Mehta',
        author_email='mehtay@janelia.hhmi.org',
        license='MIT',
        packages=[
            'plasticity'
        ],
        install_requires=[
            'imageio',
            'jax',
            'matplotlib',
            'numpy',
            'optax',
            'pandas',
            'psutil',
            'scikit-learn',
            'tqdm'
        ]
)
