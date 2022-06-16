from .loss import FocalLoss, LabelSmoothing, Adaptive_Proxy_Anchor, Proxy_NCA, MultiSimilarityLoss, ContrastiveLoss, TripletLoss, NPairLoss
from torch.nn import CrossEntropyLoss
def make_loss(config):
    if config.loss_type == 'contrastive':
        loss_func = ContrastiveLoss()
    elif config.loss_type == 'triplet':
        loss_func = TripletLoss()
    elif config.loss_type == 'npair':
        loss_func = NPairLoss()
    elif  config.loss_type == 'multisimilarity':
        loss_func = MultiSimilarityLoss()
    elif config.loss_type == 'proxy_nca':
        loss_func = Proxy_NCA(nb_classes=config.num_class, sz_embed = config.num_embeddings).to(config.device)
    elif config.loss_type == 'APA':
        loss_func = Adaptive_Proxy_Anchor(nb_classes=config.num_class, sz_embed = config.num_embeddings, config=config, mrg=0.1, alpha=32, scale_margin=1).to(config.device) 
    return loss_func
