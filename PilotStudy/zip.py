import os
import zipfile

from datetime import datetime


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

if __name__ == '__main__':
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    zipf = zipfile.ZipFile('demonstrations_{}.zip'.format(dt_string), 'w', zipfile.ZIP_DEFLATED)
    zipdir('demonstrations/', zipf)
    zipf.close()