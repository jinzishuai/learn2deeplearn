# Code
[./fl_human_policy.py](./fl_human_policy.py). One has to change the initial value to the `policy` variable.
# Results
https://gym.openai.com/envs/FrozenLake-v0/ says
> FrozenLake-v0 defines "solving" as getting average reward of 0.78 over 100 consecutive trials.

## Q-Table Learning from 20000 episons: 719 out of 1000 runs were successful
```python
policy=np.array(
        [[0, 3, 2, 3],
        [0, 0, 0, 0],
        [3, 1, 0, 0],
        [0, 2, 1, 0]]
        )

```
### Use X=-1 for Holes and Goals: 678 out of 1000 runs were successful
```python
X = -1
policy=np.array(
        [[0, 3, 2, 3],
        [0, X, 0, X],
        [3, 1, 0, X],
        [X, 2, 1, X]]
        )
```
## Human Picked Choices
It seems that the choices I manually made perform very badly. This might be because the chances of moving in my chosen direction is low.

## Best Results on OpenAMI

See [stickzman](./stickzman)
