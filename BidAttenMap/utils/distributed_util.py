import os
import torch
from torch import distributed
import torch.nn as nn
import torch.multiprocessing as mp
import logging

logger = logging.getLogger('global')

def initialSettings(port, backend='nccl'):
    method = mp.get_start_method(allow_none=True)
    if method is None:
        mp.set_start_method('spawn')
    
    logger.info('multiprocessing start method:{}'.format(method))
    procId = int(os.environ.get('SLURM_PROCID'))
    numOfTasks = int(os.environ.get('SLURM_NTASKS'))
    nodeList = os.environ.get('SLURM_JOB_NODELIST')

    numOfGPUs = torch.cuda.device_count()
    torch.cuda.set_device(procId % numOfGPUs)

    if '[' in nodeList:
        beg = nodeList.find('[')
        pos1 = nodeList.find('-', beg)
        
        if pos1 < 0:
            pos1 = 1000
        pos2 = nodeList.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        
        nodeList = nodeList[:min(pos1, pos2)].replace('[', '')
    
    addr = nodeList[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(numOfTasks)
    os.environ['RANK'] = str(procId)

    if backend == 'nccl':
        distributed.init_process_group(backend='nccl')
    else:
        distributed.init_process_group(backend='gloo', rank=procId, world_size=numOfTasks)
    
    rank = distributed.get_rank()
    worldSize = distributed.get_world_size()

    return rank, worldSize

def AverageGradients(model):
    for param in model.parameters():
        if param.requires_grad and not (param.grad is None):
            distributed.all_reduce(param.grad.data)


def broadcastParams(model):
    for pValue in model.state_dict().values():
        distributed.broadcast(pValue, 0)


class DisttributedModel(nn.Module):
    def __init__(self, model):
        super(DisttributedModel, self).__init__()
        self.model = model
        broadcastParams(self.model)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)
    
    def train(self, mode=True):
        super(DisttributedModel, self).train(mode)
        self.model.train(mode)