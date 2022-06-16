import torch

def save_checkpoint(model, optimizer, scheduler, epoch, config, model_name = '_last.pt'):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler,
            'epoch': epoch
                }, config.path_save_model+ model_name)