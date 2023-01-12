# Adaptive-Learning-rate-and-regularization-from-scratch-using-Pytorch

The math is clearer in GitHUB Repo: [Link](https://github.com/Sowmya-Iyer/Adaptive-Learning-rate-and-regularization-from-scratch-using-Pytorch)

Adaptive Learning rates: Momentum, Nesterov Momentum, and Adam; Regularization: L2

## Adaptive Learning rates implemented:

Optimizers code : [Link](optimizers.py)

### SGD

$$x_{t+1} = x_t - \alpha \nabla f(x_t)$$

```python
class SGD(Optimizer):
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
                #gt <- gt + l2.lambda * (parameter_t-1)
                gradient.data.add_(parameter,alpha=self.weight_decay)
                # Gradient Decay.
                #parameter_t = (parameter_t-1) - lr*gt
                parameter.data.add_(gradient, alpha=-self.lr)

        return None


```

### Momentum:

for each parameter $x\_t$ at the iteration $t$, and the corresponding velocity $v\_t$, we have $$v_{t+1} = \rho v_t + \alpha \nabla f(x_t)$$$$x_{t+1} = x_t - v_{t+1}$$ where $\alpha$ is the learning rate, $\rho$ is the hyperparameter for the momentum rate. A common default option for $\rho$ is $0.9$.

```python
class Momentum(Optimizer):
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
                    continue
                else:
                    gradient = parameter.grad
                #gt <- gt + l2.lambda * (parameter_t-1)
                gradient.data.add_(parameter,alpha=self.weight_decay)
                #v_(t+1) = v_t* rho + lr*g_t
                if (self.v):
                    buf = self.v[parameter]
                    # buf = buf * self.RHO
                    buf.mul_(self.RHO).add_(gradient, alpha=self.lr)
                    #saving vt+1 
                    self.v[parameter]=buf 
                    # print(buf)              
                #parameter_t = (parameter_t-1) - v_(t+1)
                parameter.data = parameter.data - buf

        return None
```

### Nesterov Momentum

for each parameter $x\_t$ at the iteration $t$, and the corresponding velocity $v\_t$, we have $$v_{t+1} = \rho v_t + \alpha \nabla f(x_t - \rho v_t)$$ $$x_{t+1} = x_t - v_{t+1}$$

```python
class Nesterov(SGD):
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
```

### Adam

for each parameter $x\_t$ at the iteration $t$, we have $$m_1 = \beta_1 * m_1 + (1-\beta_1) \nabla f(x_t)$$ $$m_2 = \beta_2 * m_2 + (1-\beta_2) (\nabla f(x_t))^2$$ $$u_1 = \frac{m_1}{1 - \beta_1^t}$$ $$u_2 = \frac{m_2}{1 - \beta_2^t}$$ $$x_{t+1} = x_t - \alpha \frac{u_1}{(\sqrt{u_2} + \epsilon)}$$ where $\alpha$ is the learning rate, $m\_1$ and $m\_2$ are the first and second moments, $u\_1$ and $u\_2$ are the first and second moments' bias correction, and $\beta\_1$, $\beta\_2$, and $\epsilon$ are hypterparameters. A set of common choices of the parameters are $\beta\_1 = 0.9$, $\beta\_2 = 0.999$, $\epsilon = 10^{-8}$. We initialize $m\_1 = m\_2 = 0$

There is a `--optim-alg` command-line option in main.py. So optimizers can be tested individually with the following commands: `python main.py --optim-alg sgd` `python main.py --optim-alg momentum` `python main.py --optim-alg nesterov` `python main.py --optim-alg adam`

```python
class Adam(Optimizer):
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
```

### Visualizing

### Loss vs Epoch

Added `--ce` argument ![plot](figure/optimizer\_ce.png)

### Accuracy vs Epoch

![plot](figure/optimizer\_acc.png)

### Observation:

* SGD with momentum and nesterov converge much faster than SGD.
* Loss is decreased with SGD with momentum and nesterov compared to SGD
* Curves of SGDs on are smoother after than ADAM because the latter has larger update value when they reach steady state

## Regularization

Code in `optimizer.py`

### L2 Regularization:

It has a hyperparameter for determining the weight of the L2 term, i.e., it is the coefficient of the L2 term called $\lambda$ which determines the weight of the regularization part.

$$L = \frac{1}{N} \sum_{i=1}^N L_i(x_i, y_i, W) + \lambda R(W)$$$$R(W) &= \sum_k \sum_j W_{k, j}^2$$

Following commands are exeuted:

```
python main.py --l2-lambda 1
python main.py --l2-lambda 0.1
python main.py --l2-lambda 0.01
```

### Visualization:

```
visualize.py --regularization
```

### Loss vs Epochs

![plot](figure/regularization\_ce.png)

### Accuracy vs epoch

![plot](figure/regularization\_acc.png)

### Observation

* for 1.0, the model underfits since the parameters add directly to gradients.
* for 0.1, the convergence is slow yet steady.
* for 0.01, the convergence is quicker, steady and optimum
