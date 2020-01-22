import numpy as np
def exp_smooth(data,alpha,steps,dones=None):
    #dones - это массив отсечек. Сквозь них не рспространяются дисконтированные награды
    data_to_mod = np.copy(data)
    for i in range(1,steps):
        roll = np.roll(data,-i)
        roll[:i+1]=np.median(data)
        if not (dones is None):
            roll[np.where(dones)[0]]=0
        data_to_mod=data_to_mod+roll*(alpha**i)
    return(data_to_mod)