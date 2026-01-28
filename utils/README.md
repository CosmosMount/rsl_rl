# utils
### activation functions
**ReLU** $$f(x)=\max(0,x)$$
**LeakyReLU** $$f(x)=\max(0.01x,x)$$
**ELU & CELU** $$f(x)=x\text{ if }x>0\text{ else }\alpha(e^x-1)$$
**SELU** $$f(x)=\lambda x \text{ if } x>0 \text{ else } \lambda\alpha(e^x-1)$$
**GELU** $$f(x)=0.5x(1+\text{erf}(x/\sqrt{2}))$$
**SiLU (Swish)** $$f(x)=x \cdot \sigma(x) = \frac{x}{1+e^{-x}}$$
**Mish** $$f(x)=x \cdot \tanh(\ln(1+e^x))$$
**Softplus** $$f(x)=\ln(1+e^x)$$
**Sigmoid** $$f(x)=\frac{1}{1+e^{-x}}$$
**Tanh** $$f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$
**Identity** $$f(x)=x$$
### optimizers
**SGD** $$\theta_{t+1} = \theta_t - \eta g_t$$
**RMSprop** $$v_t = \alpha v_{t-1} + (1-\alpha) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$$
**Adam** $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
**AdamW** $$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
### split&pad&unpad trajectories
- split trajectories with time episode
- concatentate trajectories & pad to equal-length
```
Input: [[a1, a2, a3, a4 | a5, a6],
        [b1, b2 | b3, b4, b5 | b6]]
Output:[[a1, a2, a3, a4], | [[True, True, True, True],
        [a5, a6, 0, 0],   |  [True, True, False, False],
        [b1, b2, 0, 0],   |  [True, True, False, False],
        [b3, b4, b5, 0],  |  [True, True, True, False],
        [b6, 0, 0, 0]]    |  [True, False, False, False]]
```
### resolve callable
This function enables passing custom classes or functions directly or as strings. The following formats are supported:
- Direct callable: Pass a type or function directly (e.g., MyClass, my_func)
- Qualified name with colon: "module.path:Attr.Nested" (explicit, recommended)
- Qualified name with dot: "module.path.ClassName" (implicit)
- Simple name: e.g. "PPO", "ActorCritic", ... (looks for callable in rsl_rl)