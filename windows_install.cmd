py -m pip install --upgrade pip &
py -m pip install --user virtualenv &
py -m venv venv &
Set-ExecutionPolicy Unrestricted -Scope Process &
.\venv\Scripts\activate &
pip install -r requirements_windows.txt & 
pip install -f https://github.com/Kojoley/atari-py/releases atari_py &
pip install -e .