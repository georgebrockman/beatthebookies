from setuptools import find_packages
from setuptools import setup

# with open('requirements.txt') as f:
#     content = f.readlines()
# requirements = [x.strip() for x in content if 'git+' not in x]

REQUIRED_PACKAGES = [
'pip>=9',
'setuptools>=26',
'wheel>=0.29',
'memoized-property==1.0.3',
'scikit-learn>=0.22.1',
'google-cloud-storage==1.26.0',
'gcsfs==0.6.0',
'tensorflow>=2.2',
'pandas',
'pytest',
'coverage',
'flake8',
'black',
'yapf',
'python-gitlab',
'twine',
'mlflow',
'imbalanced-learn',
'keras',
'joblib',
'streamlit']

setup(name='beatthebookies',
      # install_requires=requirements,
      install_requires=REQUIRED_PACKAGES,
      version="1.0",
      description="Beat the bookies with underdog predictions",
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/beatthebookies-run'],
      zip_safe=False)
