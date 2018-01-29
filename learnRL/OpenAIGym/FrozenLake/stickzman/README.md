# SRC

https://github.com/stickzman/honors_thesis/tree/master/Q-Value

# Results

## Q-Table (frozenlake-qmatrix.py)

The training results in a policy that is
```
qMatrix=[[ 0.22803855 -0.15200758 -0.16003714 -0.16845915]
 [-0.36       -0.38241627 -0.328       0.14745196]
 [-0.37834873 -0.4027119  -0.38013212 -0.00559167]
 [-0.41559839 -0.43670052 -0.47922178 -0.02421541]
 [ 0.2716074  -0.2        -0.2        -0.30032215]
 [ 0.          0.          0.          0.        ]
 [-0.7466272  -0.77618082 -0.0962548  -0.77370232]
 [ 0.          0.          0.          0.        ]
 [-0.2        -0.2        -0.2         0.35327095]
 [-0.2         0.45263036 -0.14332542 -0.2       ]
 [ 0.37973849 -0.47114296 -0.41504127 -0.38123081]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.56038265  0.        ]
 [ 0.          0.          0.73051412  0.        ]
 [ 0.          0.          0.          0.        ]]

Policy is [[0 3 3 3]
 [0 0 2 0]
 [3 1 0 0]
 [0 2 2 0]]
```

Test results:
```python
E:\ShiJin\learn2deeplearn\learnRL\OpenAIGym\FrozenLake>python fl_human_policy.py
[2018-01-28 21:00:52,326] Making new env: FrozenLake-v0
policy=
[[ 0  3  3  3]
 [ 0 -1  2 -1]
 [ 3  1  0 -1]
 [-1  2  1 -1]]

7416 out of 10000 runs were successful

E:\ShiJin\learn2deeplearn\learnRL\OpenAIGym\FrozenLake>
```
7391 out of 10000 runs were successful