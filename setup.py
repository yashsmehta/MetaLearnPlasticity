from setuptools import setup
import pathlib

setup(
        name='plasticity',
        version='0.1',
        description='Meta-Learning Plasticity',
        url='https://github.com/yashsmehta/plasticity',
        author='Yash Mehta',
        author_email='yashsmehta95@gmail.com',
        license='MIT',
        packages=[
            'plasticity',
            'plasticity.behavior',
            'plasticity.neural_activity',
        ],
        install_requires=pathlib.Path('requirements.txt').read_text().splitlines(),
)
