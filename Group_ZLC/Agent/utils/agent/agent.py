import torch
from utils.diffusion.sample import Planning
class Agent:
    def __init__(self, max_len: int, d_model: int, threshold: float, planning: Planning, device):
        self.max_len = max_len
        self.d_model = d_model
        self.sa = torch.randn(1, max_len, d_model).type(torch.FloatTensor).to(device)
        self.model = planning
        self.threshold = threshold
        self.t = 0
        self.alert = False
        self.device = device
    def reset(self):
        self.sa = torch.randn(1, self.max_len, self.d_model).type(torch.FloatTensor).to(self.device)
        self.t = 0
        self.alert = False
    def graph(self, sa, cur_t):
        self.sa = sa
        self.sa[:,:,19] = 0.0
        self.sa = torch.tensor(self.sa).type(torch.FloatTensor).to(self.device)
        self.t = cur_t
        pred, a = self.model.pred(self.sa, self.t)
        return pred, a
    def action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).type(torch.FloatTensor).to(self.device)
            self.sa[1, self.t, 0 : 19] = state
            if not self.alert:
                pred, a = self.model.pred(self.sa, self.t)
                self.sa = pred
                if a >= self.threshold:
                    a = 1.0
                    self.alert = True
                    ### alert begin, action always is 1
                else:
                    a = 0.0
                self.sa[1, self.t, self.d_model - 1] = a
                self.t += 1
                if self.t > self.max_len - 1:
                    self.t = self.max_len - 1
                    self.alert = True
            else:
                a = 1.0
                self.sa[1, self.t, self.d_model - 1] = a
                self.t += 1
                if self.t > self.max_len - 1:
                    self.t = self.max_len - 1
        return a