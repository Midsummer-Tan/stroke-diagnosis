import torch
import os

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_DISTRIBUTED = True if os.environ.get('WORLD_SIZE')!=None and int(os.environ.get('WORLD_SIZE'))>=2 else False