from datasets import make_dataset
from losses import make_loss
from models import make_model
from processor import train_model
from solvers import make_solver
import torch
import pandas as pd
from utils.remv_row import remv_row
from utils.logger import setup_logger
from utils.save_checkpoint import save_checkpoint
import argparse
from datasets.dataset import MVB
from datasets.make_dataset import get_augmentation
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='argparse config')
parser.add_argument('--path_model_pretrained', default = '/vinbrain/huyta/COVID_wave_4/sen/pretrain_model/densenet121_ra-50efcf5c.pth')
parser.add_argument('--path_save_log', default = './proxy_anchor_model_APA_MovingFashion/')
parser.add_argument('--resume_model', default = '/vinbrain/nguyenphan/Missing_item/proxy_anchor_model/densenet121_best.pt')
parser.add_argument('--root_data', default='/vinbrain/nguyenphan/Missing_item/')
parser.add_argument('--path_info', default="/vinbrain/nguyenphan/Missing_item/info_loss_found_APA_MovingFashion.csv")
parser.add_argument('--input_size', default=256)
parser.add_argument('--batch_size', default=60)
parser.add_argument('--num_workers', default=2)
parser.add_argument('--model_type', default='densenet121')
parser.add_argument('--device', default= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--loss_type', default='APA')
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--lr_scheduler_type', default='')
parser.add_argument('--lr_decay_step', default='30')
parser.add_argument('--lr_decay_gamma', default=0.25)
parser.add_argument('--n_epochs', default=60)
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--weight_decay', default=1e-4)
# parser.add_argument('--use_amp', default=True)
parser.add_argument('--milestones', default=(30, 50))
parser.add_argument('--num_class', default=29392)
parser.add_argument('--scale_margin', default=1)
parser.add_argument('--num_embeddings', default=512)
parser.add_argument('--is_train', default=True)
parser.add_argument('--continue_train', default=False)
config = parser.parse_args()


logger = setup_logger("Metric learning", config.path_save_log, if_train=config.is_train)
logger.info("Saving model in the path :{}".format(config.path_model_pretrained))
logger.info(config)
logger.info("Running with config:\n{}".format(config))


info = pd.read_csv(config.path_info)
info = info.sample(frac=1)

mapping = {}
index=-1
for class_id in info['class_id'].unique():
    if class_id not in mapping.keys():
        index+=1
        mapping[class_id] = index
        
        
info['class_id'] = info['class_id'].apply(lambda class_id: mapping[class_id]) 
total_class = info['class_id'].unique().shape[0]
info['phase'] = info['class_id'].apply(lambda class_id: 'train' if class_id <= total_class//2 else 'valid')
info[info['phase']=='train'].shape[0]

train = info[info['phase']=='train']
valid = info[info['phase']=='valid']
dataset = {phase: MVB(eval(phase),transform = get_augmentation(phase=phase, input_size=config.input_size)) \
        for phase in ['train','valid']}
dataloader = { phase: DataLoader(dataset=dataset[phase], num_workers=config.num_workers, batch_size=config.batch_size, \
                                          shuffle=(phase=='train'),pin_memory = (phase=='train'), drop_last=True) \
              for phase in ['train','valid'] }


loss_func = make_loss(config)
model = make_model(config).to(config.device, dtype = torch.double)

params_group = [{'params': model.parameters(), 'lr':float(config.lr)},]
if config.loss_type == 'proxy_anchor' or config.loss_type == 'proxy_nca' or config.loss_type == 'APA':
    params_group.append({'params': loss_func.proxies, 'lr':float(config.lr)*100})
    params_group.append({'params': loss_func.mrg, 'lr':float(config.lr)*1})
optimizer, lr_scheduler = make_solver(params_group, config, dataloader['train'])

if config.continue_train==True:
    print('loading status...')
    checkpoint = torch.load(config.resume_model)
    model.load_state_dict(checkpoint['model_state_dict'])

best_recall = 0
model.to(config.device)
writer = SummaryWriter(config.path_save_log)

for epoch in range(0, config.n_epochs):
    train_model(model, loss_func, dataloader['train'], optimizer, epoch, lr_scheduler, config)
    
    r_at_1 = evaluate_cos_SOP(model, dataloader['valid'])
    print('epoch: {}, recall@1: {}'.format(epoch, r_at_1))
    r_at_1 = r_at_1[0]
    save_checkpoint(model, optimizer, scheduler, epoch, config, model_name='_last.pt')
    
    if r_at_1 > best_recall:
        best_recall = r_at_1
        save_checkpoint(model, optimizer, scheduler, epoch, config, model_name='_best.pt')
    logger.info("R@1: {} at epoch {}\n".format(r_at_1, epoch))
  
    writer.add_scalar('precision@1', r_at_1, epoch)