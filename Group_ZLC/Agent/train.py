from utils.diffusion.denoise_model import Denoise
from utils.rewarder.rewarder import Rewarder
from utils.process.dataread import Data
from utils.diffusion.diffusion_train import diffusion_train
import torch
import os
EPOCH = 10000
BATCH_SIZE = 4096
SAVE_DIR = 'model'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data = Data()
denoise_model = Denoise(45, 20, 20, 2, 2, 0, device = DEVICE)
rewarder = Rewarder(45, 20, 20, 2, 2, 0, device = DEVICE)
if os.path.exists(SAVE_DIR):
    if os.path.exists(SAVE_DIR + '/' + 'diffusion.pt'):
        denoise_model = torch.load(SAVE_DIR + '/' + 'diffusion.pt')
    if os.path.exists(SAVE_DIR + '/' +'rewarder.pt'):
        rewarder = torch.load(SAVE_DIR + '/' +'rewarder.pt')
diffusion_train(EPOCH, BATCH_SIZE, SAVE_DIR, data, denoise_model, rewarder, lr = 5e-4)