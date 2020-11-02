# LearningFromHumans
Git Repo for CMPUT 656: Interactive Machine Learning pilot study. we are extending the work previously done by Seita et al. (https://arxiv.org/abs/1910.12154) to learn breakout from a curriculum of human demonstrations.

### Installation
This requires [Python 3](https://www.python.org/downloads/) to run.

We suggest you create a [Virtual Environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to run this repo.

To install dependencies (with an activated virtual environment):
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Now install the `lfh` directory
```
pip3 install -e .
```

###Acknowledgements
The majority of this code is from [Daniel Seita](https://people.eecs.berkeley.edu/~seita/) and his implementation
of his paper [ZPD Teaching Strategies for Deep Reinforcement Learning from Demonstrations
](https://arxiv.org/abs/1910.12154).