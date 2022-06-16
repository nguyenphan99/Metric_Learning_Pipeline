from .backbone import Resnet18, Resnet34, Resnet50, Resnet101, Densenet121
def make_model(config):
    if config.model_type == 'resnet18':
        model = Resnet18(config)
    elif config.model_type == 'densenet121':
        model = Densenet121(config)
    elif config.model_type == 'resnet34':
        model = Resnet34(config)
    elif config.model_type == 'resnet50':
        model = Resnet50(config)
    elif config.model_type == 'resnet101':
        model = Resnet101(config)
    return model