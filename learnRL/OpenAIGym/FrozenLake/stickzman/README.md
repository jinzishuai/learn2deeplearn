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

## Neural Network

After 2000 episods of training, we got
```
W=[array([[0.5474277 , 0.45597154, 0.4586465 , 0.45806217],
       [0.28106356, 0.26815328, 0.25643134, 0.51442456],
       [0.4311563 , 0.25800776, 0.27563262, 0.20155677],
       [0.11661144, 0.06469201, 0.06363817, 0.06752098],
       [0.5855156 , 0.4557852 , 0.43821454, 0.26257113],
       [0.05386201, 0.09014238, 0.05665828, 0.00924578],
       [0.15241857, 0.15005918, 0.16388595, 0.11103646],
       [0.03945803, 0.00645751, 0.06819009, 0.00809205],
       [0.45777702, 0.43199772, 0.4324084 , 0.608888  ],
       [0.4058982 , 0.6671269 , 0.45522636, 0.4236339 ],
       [0.5250729 , 0.33985296, 0.37220693, 0.33446705],
       [0.00840324, 0.06900503, 0.07637659, 0.06378628],
       [0.05073434, 0.06038774, 0.03725778, 0.03853187],
       [0.57626355, 0.3765114 , 0.771309  , 0.56943643],
       [0.76428777, 0.8728056 , 0.7528691 , 0.750536  ],
       [0.0564623 , 0.09944745, 0.06134057, 0.08031084]], dtype=float32)]

policy=[array([[0, 3, 0, 0],
       [0, 1, 2, 2],
       [3, 1, 0, 2],
       [1, 2, 1, 1]], dtype=int64)]

Percent of successful episodes: 47.599999999999994%


```

Testing this poilcy:
```
E:\ShiJin\learn2deeplearn\learnRL\OpenAIGym\FrozenLake>python fl_human_policy.py
[2018-01-29 15:31:15,614] Making new env: FrozenLake-v0
policy=
[[ 0  3  0  0]
 [ 0 -1  2 -1]
 [ 3  1  0 -1]
 [-1  2  1 -1]]

7343 out of 10000 runs were successful
```
Also it is proved that changing the values on the holes does not change the performance of the policy
```
E:\ShiJin\learn2deeplearn\learnRL\OpenAIGym\FrozenLake>python fl_human_policy.py
[2018-01-29 15:33:44,891] Making new env: FrozenLake-v0
policy=
[[0 3 0 0]
 [0 1 2 2]
 [3 1 0 2]
 [1 2 1 1]]

7210 out of 10000 runs were successful
```