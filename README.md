# Practical 6
Machine Learning, spring 2015

In this practical, we train an LSTM for character-level language modelling. Since this is the last week for practicals, it will be **extremely short** and does not require writing code, and is due by the end of the Friday's session (regardless of whether you are from the Wednesday or Friday session).

See PDF for details.

## Setup
Setup will be the same as last time in practical 1. Please refer to the [practical 1 repository](https://github.com/oxford-cs-ml-2015/practical1), and run the script as instructed last time. If you get an error that `nngraph` is not installed, run:
```
luarocks install nngraph
```

# Do this before reading the pdf
Clone the practical **and** download the associated data:
```
git clone https://github.com/oxford-cs-ml-2015/practical6.git
cd practical6
wget http://www.cs.ox.ac.uk/people/brendan.shillingford/teaching/practical6-data.tar.gz
tar xvf practical6-data.tar.gz
```
and start training the model:
```
th train.lua -vocabfile vocab.t7 -datafile train.t7 
```
**Make note of** the time at which you run the `train.lua` script. Every several iterations, the training script will save the current model (including its parameters) to a file called `model_autosave.t7`. You can make snapshots of this file if you want, but this is not required for the practical.

# See course page for practicals
<https://www.cs.ox.ac.uk/teaching/materials14-15/ml/>


