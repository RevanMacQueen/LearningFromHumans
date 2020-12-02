import zipfile, re, os
import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="breakout_standard", help="directory of demonstrations .zip")
    params = vars(parser.parse_args())
    extract(params["dir"])