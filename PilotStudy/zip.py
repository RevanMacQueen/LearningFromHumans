import os
import zipfile

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

if __name__ == '__main__':
    zipf = zipfile.ZipFile('demonstrations.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('demonstrations/', zipf)
    zipf.close()