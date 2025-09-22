# PRoP: Private and Personalized Robot Policies

Here we include the code for our recent work: **Fine-Tuning Robot Policies While Maintaining User Privacy**. 

## Behavior Cloning

Run `python -m bc.main`. Options are:

```bash
usage: main.py [-h] [--dim DIM] [--no-print] [--save-to-file] [--dirname DIRNAME] [--lr LR]
               [--epochs EPOCHS] [--N-datapoints N_DATAPOINTS] [--keysize KEYSIZE]

optional arguments:
  -h, --help            show this help message and exit
  --dim DIM
  --no-print
  --save-to-file
  --dirname DIRNAME
  --lr LR
  --epochs EPOCHS
  --N-datapoints N_DATAPOINTS
  --keysize KEYSIZE
```

## Reinforcement Learning

Run `python -m ppo.main`. Options are:

```bash
usage: main.py [-h] [--filename FILENAME] [--epochs EPOCHS] [--keysize KEYSIZE] [--no-print] [--plot]

optional arguments:
  -h, --help           show this help message and exit
  --filename FILENAME
  --epochs EPOCHS
  --keysize KEYSIZE
  --no-print
  --plot
```

## Image Classification

Run `python -m mnist.main`. Options are:

```bash
usage: main.py [-h] [--no-print] [--save-to-file] [--dirname DIRNAME] [--lr LR] [--epochs EPOCHS]
               [--N-datapoints N_DATAPOINTS] [--keysize KEYSIZE]

optional arguments:
  -h, --help            show this help message and exit
  --no-print
  --save-to-file
  --dirname DIRNAME
  --lr LR
  --epochs EPOCHS
  --N-datapoints N_DATAPOINTS
  --keysize KEYSIZE
```

