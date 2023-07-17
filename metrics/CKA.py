import torch


def CKA_function(X, Y):
    assert X.shape[0] == Y.shape[0]
    X = X - torch.mean(X, dim=0)
    Y = Y - torch.mean(Y, dim=0)

    XTX = X.t() @ X
    YTY = Y.t() @ Y
    XTY = X.t() @ Y

    result = XTY.norm(p="fro") ** 2 / (XTX.norm(p="fro") * YTY.norm(p="fro"))

    del XTX, YTY, XTY
    torch.cuda.empty_cache()

    return result
