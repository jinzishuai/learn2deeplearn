Reference https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

# [FrozenLake Environment in OpenAI gym](https://gym.openai.com/envs/FrozenLake-v0/)

## Initial Condition

```
SFFF
FHFH
FFFH
HFFG
```

Note that this initial condition does not change between runs (ie, identical after every `env.reset()`).
This problem does NOT require a solution policy that can deal with any initial condition. 

## Q-Table
```
Q = np.zeros([env.observation_space.n,env.action_space.n])
```
it is of size 16 (number of cells) x 4 (number of actions).

|State: Cell| Action 0: Left|Action 1: Down|Action 2: Right|Action 3: Up|
|--|--|--|--|--|
|0: (0,0)| | | | |
|1: (0,1)| | | | |
|2: (0,2)| | | | |
|3: (0,3)| | | | |
|4: (1,0)| | | | |
...
|15: (3, 3)| | | | |

### Q-Value Update 
```
Q(s,a) = r + γ(max(Q(s’,a’)))
```
* r: reward of current state.  The reward at every step is 0, except for entering the goal, which provides a reward of 1.

#### The Policy

The policy would be based on the Q-table: `np.argmax(Q,axis=1).reshape(4,4)`
#### Human Generated Policy
```
|1/2| 2 | 2 | 0 | 
| 1 | X | 1 | X | 
| 2 |1/2| 1 | X | 
| X | 2 | 2 | X |
```
X means it does not matter.
Note that this policy only works for the "non-slippery" setup. In this original problem, it is a very BAD policy. See https://github.com/jinzishuai/learn2deeplearn/issues/40 for details.

## stickzman's code as the best solution in OpenAI so far

https://github.com/stickzman/honors_thesis/tree/master/Q-Value
