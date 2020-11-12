py -m pip install --upgrade pip &
py -m pip install --user virtualenv &
py -m venv venv &
venv\Scripts\Activate.bat &
pip install -r requirements_windows.txt & 
pip install -f https://github.com/Kojoley/atari-py/releases atari_py &
pip install -e .