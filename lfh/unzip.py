import zipfile, re, os

import os
import io
import zipfile

def extract(filename):
    z = zipfile.ZipFile(filename)
    for f in z.namelist():
        print(f)
        # get directory name from file
        dirname = os.path.splitext(f)[0]  
        # create new directory
        print(dirname)
        os.makedirs(dirname)  
        



        # read inner zip file into bytes buffer 
        content = io.BytesIO(z.read(f))
        zip_file = zipfile.ZipFile(content)
        for i in zip_file.namelist():
            zip_file.extract(i, dirname)

extract("../BreakoutDemonstrations-20201118T035523Z-001.zip")