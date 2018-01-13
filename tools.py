import numpy as np

def LinearCase(Outputs, W_w, W_b, d_prime):
    k = W_w.shape[0]
    c = W_w.shape[2]
    d = W_w.shape[3]
    n = Outputs[0].shape[0]

    Y = Outputs[0].reshape(n*n,d).transpose()
    for i in range(1,len(Outputs)):
        Y = np.concatenate((Y,Outputs[i].reshape(n*n,d).transpose()),axis=1)
    Y_bar = np.mean(Y,axis=1,keepdims=True)
    Y = Y - Y_bar

    Cov = Y.dot(Y.transpose())
    w, v = np.linalg.eig(Cov)
    M = v[:d_prime].transpose().dot(v[:d_prime])
    P = Q = v[:d_prime].transpose()
    
    W = np.concatenate((W_w.reshape(k*k*c,d),W_b.reshape(1,d))).transpose()
    W_prime = Q.transpose().dot(W)
    
    b = (Y_bar - M.dot(Y_bar)).reshape(d)
    
    weight_conv1_W = W_prime[:,:(k*k*c)].transpose().reshape(k,k,c,d_prime)
    weight_conv1_b = W_prime[:,(k*k*c)]
    weight_conv2_W = P.transpose().reshape(1,1,d_prime,d)
    weight_conv2_b = b
    
    return weight_conv1_W, weight_conv1_b, weight_conv2_W, weight_conv2_b