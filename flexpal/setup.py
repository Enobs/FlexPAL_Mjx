from setuptools import setup, find_packages

setup(
    name='softmujoco',
    version='0.1.0',
    author='Yinan MENG',
    author_email='yinan.meng@manchester.ac.uk',
    description='A custom Gym environment for a 3D robot with 3 serial joints',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # replace with the real url of your package
    url='https://github.com/enobs/3d_robot_gym_env',
    packages=find_packages(),
    install_requires=[
        'gym>=0.17.0',
        'numpy>=1.19.5',
        'scipy>=1.5.4',
        'matplotlib>=3.3.4',
        'torch>=1.8.1',
        'scikit-learn>=0.24.1',
        'pandas>=1.2.3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
