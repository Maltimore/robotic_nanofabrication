import torch


def get_torch_optimizer(model, params):
    try:
        if params['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params['lr'],
                momentum=params['momentum'])
        elif params['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['lr'])
        return optimizer
    except ValueError:
        # occurrs when model.parameters() is empty
        return None
