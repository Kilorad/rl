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

def cosine_similarity(X,Y):
    #X,Y - матрицы. axis 0 - это номер вектора, axis 1 - координаты вектора
    prod_mx = X*Y
    X_modules = np.sqrt(np.sum(X**2,axis=0))
    Y_modules = np.sqrt(np.sum(Y**2,axis=0))
    return np.sum(prod_mx / (X_modules*Y_modules) ,axis=0)