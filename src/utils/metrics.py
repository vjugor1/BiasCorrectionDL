import torch   


def float_to_score(float_tensor, thresh=20):
    float_tensor = torch.div(float_tensor, thresh+10)
    clipped_tensor = torch.clip(float_tensor, min=0, max=0.999)
    return clipped_tensor


def float_to_binary(float_tensor, thresh=20):
    float_tensor = torch.div(float_tensor, thresh)
    float_tensor = torch.floor(float_tensor)
    clipped_tensor = torch.clip(float_tensor, min=0, max=1)
    return clipped_tensor.int()


def get_outliers_s(predictions, target, thresh=20):
    """ 
    Return target values where value is higher than threshold, 
    and respective predictions
    """
    target = torch.where(target > thresh, target, 0.)
    indices = torch.nonzero(target)[:,0]
    target = torch.index_select(target, 0, indices)
    predictions = torch.index_select(predictions, 0, indices)

    return predictions, target


def get_outliers_p(predictions, target, thresh=20):
    """ 
    Return predictions where value is higher than threshold, 
    and respective target values
    """

    predictions = torch.where(predictions > thresh, predictions, 0.)
    indices = torch.nonzero(predictions)[:,0]
    target = torch.index_select(target, 0, indices)
    predictions = torch.index_select(predictions, 0, indices)

    return predictions, target