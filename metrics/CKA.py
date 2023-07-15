import torch as t

def CKA_function(X, Y):
    assert X.shape[0] == Y.shape[0]
    # center along the zero dim
    X = X - t.mean(X, dim=0)
    Y = Y - t.mean(Y, dim=0)

    return t.norm(X.t() @ Y, p="fro") ** 2 / ((t.norm(X.t() @ X, p = "fro")) * (t.norm(Y.t() @ Y, p = "fro")) )
