#
import torch
from typing import Iterable, Optional, Callable
from collections import defaultdict

class Optimizer(torch.optim.Optimizer):
    r"""
    Optimizer.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        # Save necessary attributes.
        self.lr = lr
        self.weight_decay = weight_decay

        # Super call.
        torch.optim.Optimizer.__init__(self, parameters, dict())

    @torch.no_grad()
    def prev(self, /) -> None:
        r"""
        Operations before compute the gradient.
        PyTorch has design problem of compute Nesterov SGD gradient.
        PyTorch team avoid this problem by using an approximation of Nesterov
        SGD gradient.
        Also, using closure can also solve the problem, but it maybe a bit
        complicated for this homework.
        In our case, function is provided as auxiliary function for simplicity.
        It is called before `.backward()`.
        This function is only used for Nesterov SGD gradient.
        """
        # Do nothing.
        pass

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        #
        ...


class SGD(Optimizer):
    r"""
    SGD.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        #
        Optimizer.__init__(self, parameters, lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        # Traverse parameters of each groups.
        for group in self.param_groups:
            #  
            for parameter in group['params']:
                # Get gradient without weight decaying.
                if parameter.grad is None:
                    #
                    continue
                else:
                    #
                    gradient = parameter.grad

                # Apply weight decay.
                # YOU SHOULD FILL IN THIS FUNCTION
                ...
                #gt <- gt + l2.lambda * (parameter_t-1)
                gradient.data.add_(parameter,alpha=self.weight_decay)
                # Gradient Decay.
                #parameter_t = (parameter_t-1) - lr*gt
                parameter.data.add_(gradient, alpha=-self.lr)

        return None


class Momentum(Optimizer):
    R"""
    Momentum.
    """
    # YOU SHOULD FILL IN THIS FUNCTION
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
    
        Optimizer.__init__(self, parameters, lr=lr, weight_decay=weight_decay)
        self.RHO = 0.9
        self.v={}

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        if(bool(self.v)==False):
            for group in self.param_groups:
                for parameter in group['params']:
                    #setting v1
                    self.v[parameter] =  torch.zeros_like(parameter.data)
        for group in self.param_groups:
            for parameter in group['params']:
                # Get gradient without weight decaying.
                if parameter.grad is None:
                    #
                    continue
                else:
                    #
                    gradient = parameter.grad
                #gt <- gt + l2.lambda * (parameter_t-1)
                gradient.data.add_(parameter,alpha=self.weight_decay)
                #v_(t+1) = v_t* rho + lr*g_t
                if (self.v):
                    buf = self.v[parameter]
                    # buf = buf * self.RHO
                    # print("-----------------------------------------------")
                    buf.mul_(self.RHO).add_(gradient, alpha=self.lr)
                    #saving vt+1 
                    self.v[parameter]=buf 
                    # print(buf)              
                #parameter_t = (parameter_t-1) - v_(t+1)
                parameter.data = parameter.data - buf

        return None

class Nesterov(SGD):
    R"""
    Nesterov.
    """
    # YOU SHOULD FILL IN THIS FUNCTION
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
    
        Optimizer.__init__(self, parameters, lr=lr, weight_decay=weight_decay)
        self.RHO = 0.9
        self.v={}
        # self.g={}

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        if(bool(self.v)==False):
            for group in self.param_groups:
                for parameter in group['params']:
                    #setting v1
                    #if nesterov
                    # v = gradient 
                    self.v[parameter] =  torch.zeros_like(parameter.data)
                    # self.g[parameter]= torch.zeros_like(parameter.data)
        for group in self.param_groups:
            for parameter in group['params']:
                # Get gradient without weight decaying.
                if parameter.grad is None:
                    #
                    continue
                else:
                    #
                    gradient = parameter.grad
                #gt <- gt + l2.lambda * (parameter_t-1)
                gradient.data.add_(parameter,alpha=self.weight_decay)
                #v_(t+1) = v_t* rho + lr*g_t
                # if (self.v):
                #     buf = self.v[parameter]
                #     # buf = buf * self.RHO
                #     # print("-----------------------------------------------")
                #     buf.mul_(self.RHO).add_(gradient)
                #     #saving vt+1 
                #     self.v[parameter]=buf

                #     gradient =  self.g[parameter].add(buf,alpha=self.RHO)
                #     self.g[parameter] = gradient
                    # print(buf)     
                if (self.v):
                    buf = self.v[parameter]
                    # buf = buf * self.RHO
                    # print("-----------------------------------------------")
                    
                    buf.mul_(self.RHO)
                    gradient.add_(buf,alpha=-1)
                    buf.add_(gradient, alpha=self.lr)
                    #saving vt+1 
                    self.v[parameter]=buf     
                #parameter_t = (parameter_t-1) - v_(t+1)
                parameter.data = parameter.data - buf

        return None

class Adam(Optimizer):
    R"""
    Adam.
    """
    #
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
    
        Optimizer.__init__(self, parameters, lr=lr, weight_decay=weight_decay)
        self.RHO = 0.9
        self.BETA1 = 0.9
        self.BETA2 = 0.999
        self.EPSILON = 1e-8
        self.avg={}
        self.avg_sq={}
        self.state={}

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        if(bool(self.avg)==False or bool(self.avg_sq)==False):
            for group in self.param_groups:
                for parameter in group['params']:
                    #setting v1
                    #if nesterov
                    # v = gradient 
                    self.avg[parameter] = torch.zeros_like(parameter.data)
                    self.avg_sq[parameter] =torch.zeros_like(parameter.data)
                    self.state[parameter] =1
        for group in self.param_groups:
            for parameter in group['params']:                
                #gt <- gt + l2.lambda * (parameter_t-1)
                # #v_(t+1) = v_t* rho + lr*g_(parameter -vt)
                # if (self.state[parameter]==0):
                #     self.state[parameter]+=1
                if (self.avg and self.avg_sq):
                    gradient= parameter.grad
                    gradient.data.add_(parameter,alpha=self.weight_decay)
                    gradient_squared = torch.square(gradient)

                    m1 = self.avg[parameter]
                    m2 = self.avg_sq[parameter]
                    # buf = buf * self.RHO
                    
                    m1.mul_(self.BETA1).add_(gradient, alpha=(1-self.BETA1))
                    m2.mul_(self.BETA2).add_(gradient_squared, alpha=(1-self.BETA2)) 
                    #saving vt+1 
                    u1=torch.div(m1,(1-self.BETA1**self.state[parameter]))
                   
                    u2=torch.div(m2,(1-self.BETA2**self.state[parameter]))
                    self.state[parameter]+=1
                den = torch.sqrt(u2) + self.EPSILON 
                num = torch.mul(u1,self.lr) 
                parameter.data = parameter.data - (num/den)

        return None
    # YOU SHOULD FILL IN THIS FUNCTION
    ...