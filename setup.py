from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = [
    line.strip()
    for line in (ROOT / 'requirements.txt').read_text(
        encoding='utf-8'
    ).splitlines()
    if line.strip() and not line.lstrip().startswith('#')
]


setup(
    name='frequency-guided-restoration',
    version='0.1.0',
    description='Frequency-guided multi-range video restoration on BasicSR',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'basicsr.metrics': ['*.npz'],
        'basicsr.ops.dcn': ['src/*'],
    },
    python_requires='>=3.8',
    install_requires=REQUIREMENTS,
    license='Apache-2.0',
)
