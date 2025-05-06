def compute_iou(pred, true):
    inter = np.logical_and(pred==true, True).sum()
    union = np.logical_or(pred==true, True).sum()
    return inter/(union+1e-9)