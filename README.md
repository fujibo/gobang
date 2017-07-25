# Gobang

This is a program that a computer learns to play gobang. To learn how it plays gobang, DQN is used.

## How To Use
* if you want to play with computer with GUI
```bash
python gui.py
```
* if you want to play with computer without GUI
```bash
python test.py
```
* if you want to train this AI
```bash
python main.py
```

## Assumed Environment
* python3 (anaconda)
* chainer1.24
* opencv
* numba

```bash
# to install chainer 
pip install chainer==1.24.0
# to install opencv3
conda install -c menpo opencv3
```

## About Source Code
* main.py
  * for training
* test.py
  * for playing with this AI without GUI
* gui.py
  * for playing with this AI with GUI
* evaluation.py
  * a part of neural network in this program
* main_play.py
  * a trash code
