import numpy as np
def exp_smooth(data,alpha,steps):
    data_to_mod = np.copy(data)
    for i in range(1,steps):
        roll = np.roll(data,-i)
        roll[:i+1]=np.median(data)
        data_to_mod=data_to_mod+roll*(alpha**i)
    return(data_to_mod)