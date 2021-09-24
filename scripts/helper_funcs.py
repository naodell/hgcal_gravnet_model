
from collections import OrderedDict

def convert_parallel_model(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    # load params
    return new_state_dict
