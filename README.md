# LearningFromHumans
Git Repo for CMPUT 656: Interactive Machine Learning pilot study. we are extending the work previously done by Seita et al. (https://arxiv.org/abs/1910.12154) to learn breakout from a curriculum of human demonstrations.

## Installation

### Linux/MacOS
This requires [Python 3](https://www.python.org/downloads/) to run.

We suggest you create a [Virtual Environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to run this repo.

First clone this repository.
```
git clone https://github.com/RevanMacQueen/LearningFromHumans.git
cd LearningFromHumans
```

To install dependencies (with an activated virtual environment):
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Now install the `lfh` directory
```
pip3 install -e .
```

### Windows
This requires [Python 3](https://www.python.org/downloads/) to run.

We suggest you create a [Virtual Environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to run this repo.

First clone this repository.
```
git clone https://github.com/RevanMacQueen/LearningFromHumans.git
cd LearningFromHumans
```

To install dependencies (with an activated virtual environment):
```
pip install --upgrade pip
pip install -r requirements_windows.txt
pip install -f https://github.com/Kojoley/atari-py/releases atari_py 
```
Now install the `lfh` directory
```
pip install -e .
```

## Pilot Study Instructions

Thanks for taking the time to be part of our pilot study! You will be helping a reinforcement learning agent learn how to play the Atari game Breakout. If you're unfamiliar with this game, please check out this [video](https://www.youtube.com/watch?v=V1eYniJ0Rnk). You will play Breakout and the agent will learn how to play based off of your demonstrations. After installing (see above) follow these steps to contribute to the pilot study. (Please read all steps first) 

### 1. Change directory into the PilotStudy folder.

```cd PilotStudy```

### 2. Run the PilotStudy File.

Once in the PilotStudy directory, run the following command to begin the pilot study if you're running MacOS or Linux: 
```
python3 main.py
```

or if you're running Windows:
```
python main.py
```

Here you will be given on-screen instuctions how to complete the pilot study. The UI is somewhat finicky, you'll receive prompts via the terminal and to play Breakout a separate window will appear. You may need to click on the window before it will accept keyboard input. The PilotStudy file will prompt you to keep play more games, or to exit. If you exit, you have the option to run it again to contribute more demonstrations :)  

### 3. Zip up demonstrations.

Once you're done giving demonstrations, run the following command to zip up all your demonstrations:
```
python3 zip.py # MacOS or Linux
```

```
python main.py # Windows
```


This will create a new file demonstrations.zip in the PilotStudy directory

### 4. Send us the demonstrations.

Upload your demonstrations.zip to this [google drive](https://drive.google.com/drive/folders/1ZvrUFTViP6u3XR2V1wuE_JLINQ7cqYlY?usp=sharing)


## Issues

### Installing atari-py on Windows 10
There are some problems installing atari-py on Windows 10. Make sure you set up a virtual environment and install dependencies from the requirements_windows.txt. If things still aren't working let us know and we'll try to figure things out.

## Acknowledgements
The majority of this code is from [Daniel Seita](https://people.eecs.berkeley.edu/~seita/) and his implementation
of his paper [ZPD Teaching Strategies for Deep Reinforcement Learning from Demonstrations
](https://arxiv.org/abs/1910.12154).

