import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
	
setuptools.setup(
     name='rosebud',  
     version='0.1',
     scripts=['rosebud.py'] ,
	 install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'missingno'],
     author="Ralph Puzon",
     author_email="puzonralph@gmail.com",
     description="A python data exploration and analytics package",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/ralphpuzon/rosebud",
     packages=setuptools.find_packages(),
	 keywords=['data', 'data visualization','data exploration', 'data analysis', 'missing data', 'data science', 'pandas', 'python',
              'jupyter'],
     classifiers=[],
 )