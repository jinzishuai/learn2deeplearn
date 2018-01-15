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
