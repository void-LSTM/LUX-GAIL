import array as np
def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state 

dict1={'s':12,'sd':123}
print(type(dict1))
print(type(from_json(dict1)))