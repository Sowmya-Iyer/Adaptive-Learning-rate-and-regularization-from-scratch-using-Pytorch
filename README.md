# Adaptive-Learning-rate-and-regularization-from-scratch-using-Pytorch
Adaptive Learning rates: Momentum, Nesterov Momentum, and Adam; Regularization: L2

## Adaptive Learning rates implemented:
### SGD
$$x_{t+1} = x_t - \alpha \nabla f(x_t)$$

### Momentum:
for each parameter $x_t$ at the iteration $t$, and the corresponding velocity $v_t$, we have
$$v_{t+1} = \rho v_t + \alpha \nabla f(x_t) $$
$$x_{t+1} = x_t -  v_{t+1}$$
where $\alpha$ is the learning rate, $\rho$ is the hyperparameter for the momentum rate. A common default option for $\rho$ is $0.9$.

### Nesterov Momentum
for each parameter $x_t$ at the iteration $t$, and the corresponding velocity $v_t$, we have
$$v_{t+1} = \rho v_t + \alpha  \nabla f(x_t - \rho v_t)$$
$$x_{t+1} = x_t - v_{t+1}$$

### Adam
for each parameter $x_t$ at the iteration $t$, we have
$$m_1 = \beta_1 * m_1 + (1-\beta_1) \nabla f(x_t)$$
$$m_2 = \beta_2 * m_2 + (1-\beta_2) (\nabla f(x_t))^2$$
$$u_1 = \frac{m_1}{1 - \beta_1^t}$$
$$u_2 = \frac{m_2}{1 - \beta_2^t}$$
$$x_{t+1} = x_t - \alpha \frac{u_1}{(\sqrt{u_2} + \epsilon)}$$
where $\alpha$ is the learning rate, $m_1$ and $m_2$ are the first and second moments, $u_1$ and $u_2$ are the first and second moments' bias  correction, and $\beta_1$, $\beta_2$, and $\epsilon$ are hypterparameters. A set of common choices of the parameters are $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. We initialize $m_1 = m_2 = 0$


There is a `--optim-alg` command-line option in main.py. So optimizers can be tested individually with the following commands:
    `python main.py --optim-alg sgd`
    `python main.py --optim-alg momentum`
    `python main.py --optim-alg nesterov`
    `python main.py --optim-alg adam`
  
### Visualizing
### Loss vs Epoch
Added `--ce` argument 
![plot](https://github.com/Sowmya-Iyer/Adaptive-Learning-rate-and-regularization-from-scratch-using-Pytorch/blob/main/figure/optimizer_ce.png)

### Accuracy vs Epoch

![plot](https://github.com/Sowmya-Iyer/Adaptive-Learning-rate-and-regularization-from-scratch-using-Pytorch/blob/main/figure/optimizer_acc.png)

### Observation:
- SGD with momentum and nesterov converge much faster than SGD.
- Loss is decreased with SGD with momentum and nesterov compared to SGD
- Curves of SGDs on are smoother after than ADAM because the latter has larger update value
when they reach steady state

## Regularization

Code in `optimizer.py` 

### L2 Regularization:
It has a hyperparameter for determining the weight of the L2 term, i.e., it is the coefficient of the L2 term called $\lambda$ which determines the weight of the regularization part.

$$L = \frac{1}{N} \sum_{i=1}^N L_i(x_i, y_i, W) + \lambda R(W) $$
$$R(W) &= \sum_k \sum_j W_{k, j}^2$$

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
![plot](https://github.com/Sowmya-Iyer/Adaptive-Learning-rate-and-regularization-from-scratch-using-Pytorch/blob/main/figure/regularization_ce.png)

### Accuracy vs epoch
![plot](https://github.com/Sowmya-Iyer/Adaptive-Learning-rate-and-regularization-from-scratch-using-Pytorch/blob/main/figure/regularization_acc.png)

### Observation
- for 1.0, the model underfits since the parameters add directly to gradients.
- for 0.1, the convergence is slow yet steady.
- for 0.01, the convergence is quicker, steady and optimum
