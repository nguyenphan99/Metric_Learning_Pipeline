from torch import nn, optim
import torch

def make_solver(params_group, config, train_dataloader, ):
    
    if config.optimizer =='sgd':
        optimizer = torch.optim.SGD(params_group, lr=float(config.lr), weight_decay = config.weight_decay, momentum = 0.9, nesterov=True)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_group, lr=float(config.lr), weight_decay = config.weight_decay)
    elif config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params_group, lr=float(config.lr), alpha=0.9, weight_decay = config.weight_decay, momentum = 0.9)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params_group, lr=float(config.lr), weight_decay = config.weight_decay)
        #optimizer = optim.Adam(model.parameters(), lr=lr)
    if config.lr_scheduler_type == 'onecyclelr':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=config.n_epochs)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config.lr_decay_step, gamma=config.lr_decay_gamma)
    return optimizer, lr_scheduler