import torch
import torch.nn as nn
import os

        
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = self.fc4(x)
        return x
        
class PGDAttack_ViT(object):
    """ 
        Attack parameter initialization. The attack performs k steps of size 
        alpha, while always staying within epsilon from the initial point.
            IFGSM(Iterative Fast Gradient Sign Method) is essentially 
            PGD(Projected Gradient Descent) 
    """
    def __init__(self, device, inception, epsilon=0.1, k=40, alpha=0.01, random_start=False):
        self.device = device
        self.inception_model = inception

        self.scorer = FNN()
        self.scorer.load_state_dict(torch.load(THE_PATH_TO_YOUR_DENSITY_ESTIMATOR))
        self.scorer.to(device)

        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.random_start = random_start

        
    def get_score(self, imgs, no_grad=True): 
        if no_grad:
            with torch.no_grad():
                return self.scorer(self.inception_model(imgs, detach=True)).squeeze()
        else:
            return self.scorer(self.inception_model(imgs, detach=False)).squeeze()
        
        
    def __call__(self, x, gen_c, sync, run_G, reverse=False, specified_k=None):
        if self.random_start:
            x_adv = x + x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else: 
            x_adv = x.clone()
        
        K = self.k if specified_k == None else specified_k
        for i in range(K):
            x_adv.requires_grad_()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            
            img, _gen_ws = run_G(x_adv, gen_c, sync=sync)

            obj = - self.get_score(img, no_grad=False)
            if reverse:
                obj = -obj  # Simply negate the entire objective
            obj = obj.mean()
            obj.backward()
            grad = x_adv.grad.clone()

            # update x_adv
            x_adv = x_adv.detach() + self.alpha * grad.sign()       # decrease quality score
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)

        return x_adv.detach()
        
        
    def attack_random(self, x, gen_c, sync, run_G, run_D, reverse=False, specified_k=None):
        if self.random_start:
            x_adv = x + x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else: 
            x_adv = x.clone()
        
        K = self.k if specified_k == None else specified_k
        for i in range(K):
            x_adv.requires_grad_()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            
            img, _gen_ws = run_G(x_adv, gen_c, sync=sync)

            obj = - run_D(img)
            if reverse:
                obj = -obj  # Simply negate the entire objective
            obj = obj.mean()
            obj.backward()
            grad = x_adv.grad.clone() * ((torch.rand_like(x_adv) > 0.5).float() * 2 - 1)  # random direction

            # update x_adv
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)

        return x_adv.detach()