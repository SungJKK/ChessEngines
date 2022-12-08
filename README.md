# Chess engines project
(New) Repository for AMS 325 final project.  
- Note: severe, complicated issues in github due to exceeding file size limits arose due to `.h5`files (for saved DQN models), so I debunked previously created repo and just copied the files into a new repo. Though large files that exceed the Github upload limits can be uploaded using [`git-lfs`](https://git-lfs.github.com/), files were still to large and issues continuously emerged. Thus to use the DQN model, one must run the training on their local machine first using instructions [here](#creating-dqn-model-file).
- For whatever reason if curious, the previous repository is archived
  [here](https://github.com/SungJKK/chess_engines).

# Table of Contents
- [Description](#description)
- [Running Files](#running-files)
- [Analysis of Result](#analysis-of-results)
- [Authors](#authors)
- [References](#references)


# Description
Implementing 2 chess engines:
1. Naive algorithm
    - datasets acquired from [lichess](https://database.lichess.org/)
    - optional dataset: [chess.com](https://www.chess.com/news/view/published-data-api#pubapi-endpoint-games-archive-list)
2. Deep learning neural network algorithm


# Running Files 
### Setting up the environment
```sh
$ git clone https://github.com/SungJKK/chess_engines && cd chess_engines
$ conda env create -n chess_engines -f environment.yml
$ conda activate chess_engines
```
- Note: if you are using mac arm-processors, follow the steps below or follow [this guide](https://stackoverflow.com/questions/72964800/what-is-the-proper-way-to-install-tensorflow-on-apple-m1-in-2022).
```sh
$ git clone https://github.com/SungJKK/chess_engines && cd chess_engines
$ CONDA_SUBDIR=osx-arm64 conda env create -n chess_engines -f environment.yml
$ conda activate chess_engines
$ conda config --env --set subdir osx-arm64
```

### Creating DQN model file
- Note: due to file size limit of github, h5 files could not be uploaded. To use the DQN chess
  engine, please setup local environment by following above instructions and create the trained
  model by running `train.py`.
```sh
$ python train.py
```
- This will create 2 .h5 files, q_network.h5 and target_network.h5, inside `_saved_models/{version}`
  based on the version set in `train.py` file.
    - v0.0: trained for 1,000 iterations 
    - v0.1: trained for 5,000 iterations
    - v0.2: trained for 10,000 iterations


### Running the engines
- To see in real time how the engines compete against each other, run the main file.
```sh
$ python main.py
```


# Analysis of Results
- See [ChessAnalysisP2.ipynb](_notebooks/ChessAnalysisP2.ipynb)


# Authors
[Sung Joong Kim](https://github.com/SungJKK) and [Bernard Tenreiro](https://github.com/BernardTenreiro)


# References
- [python-chess library](https://python-chess.readthedocs.io/en/latest/)
- [Minimax algorithm](https://en.wikipedia.org/wiki/Minimax)
- [Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [Simplified Evaluation Function](https://www.chessprogramming.org/Simplified_Evaluation_Function)
- [Can deep reinforcement learning solve chess?](https://towardsdatascience.com/can-deep-reinforcement-learning-solve-chess-b9f52855cd1e)
- [Deep q learning with Tensorflow](https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/)
- [AlphaZero paper](https://doi.org/10.48550/arXiv.1712.01815)

