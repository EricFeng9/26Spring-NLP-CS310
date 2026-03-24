# CS310 Natural Language Processing
## Assignment 1 (written)

**Total points: 10**

Consider a softmax layer of output size $k (k>1)$. Within unit $i (i=1, \dots, k)$, the linear before activation is $z_i$ (also called logits), and the output after activation is $o_i$. We have the following equation for forward propagation:
$$o_i = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)}$$
The cross-entropy loss between an output prediction $\hat{y} = [o_1, \dots, o_k]$ and the ground truth label $y = [y_1, \dots, y_k]$ (one-hot encoded) is defined as:
$$L_{CE} = -\sum_{i=1}^k y_i \log(\hat{y}_i) = -\sum_{i=1}^k y_i \log o_i$$

---

### Question 1
Show that when $k=2$, the loss is equivalent to binary cross-entropy (BCE) loss $L_{BCE} = -y_i \log \hat{y}_i - (1-y_i) \log(1-\hat{y}_i)$.
(4 points, show your math derivation to get the credits)

【Answer】
For $k=2$, the output prediction is $\hat{y} = [o_1, o_2]$ and the ground truth label is $y = [y_1, y_2]$.
Because $y$ is one-hot encoded, we have $y_1 + y_2 = 1$, which means $y_2 = 1 - y_1$.
Because of the softmax, we have $o_1 + o_2 = 1$, which means $o_2 = 1 - o_1$.

The cross-entropy loss for $k=2$ is:
$$L_{CE} = -(y_1 \log o_1 + y_2 \log o_2)$$

Substituting $y_2 = 1 - y_1$ and $o_2 = 1 - o_1$:
$$L_{CE} = -(y_1 \log o_1 + (1 - y_1) \log (1 - o_1))$$
$$L_{CE} = -y_1 \log o_1 - (1 - y_1) \log (1 - o_1)$$

If we let $y_i$ denote the ground truth for class 1 and $\hat{y}_i$ denote the predicted probability $o_1$:
$$L = -y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i)$$
This is BCE loss formula.

---

### Question 2
What is the gradient w.r.t the $i$-th unit $\frac{\partial L}{\partial z_i}$?
(6 points, show your math derivation to get the credits)

【Answer】
The loss function is $L = -\sum_{j=1}^k y_j \log o_j$.

First, compute $\frac{\partial L}{\partial o_j}$:
$$\frac{\partial L}{\partial o_j} = -\frac{y_j}{o_j}$$

Next, compute the derivative of the softmax function $\frac{\partial o_j}{\partial z_i}$:
- If $j = i$:
$$\frac{\partial o_i}{\partial z_i} = \frac{\exp(z_i) \sum_{m} \exp(z_m) - \exp(z_i) \exp(z_i)}{(\sum_{m} \exp(z_m))^2} = o_i(1 - o_i)$$
- If $j \neq i$:
$$\frac{\partial o_j}{\partial z_i} = \frac{0 \cdot \sum_{m} \exp(z_m) - \exp(z_j) \exp(z_i)}{(\sum_{m} \exp(z_m))^2} = -o_j o_i$$

substitute these back into the gradient equation:
$$\frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial o_i} \frac{\partial o_i}{\partial z_i} + \sum_{j \neq i} \frac{\partial L}{\partial o_j} \frac{\partial o_j}{\partial z_i}$$
$$\frac{\partial L}{\partial z_i} = -\frac{y_i}{o_i} [o_i(1 - o_i)] + \sum_{j \neq i} \left( -\frac{y_j}{o_j} \right) (-o_j o_i)$$
$$\frac{\partial L}{\partial z_i} = -y_i(1 - o_i) + \sum_{j \neq i} y_j o_i$$
$$\frac{\partial L}{\partial z_i} = -y_i + y_i o_i + o_i \sum_{j \neq i} y_j$$

Since it is one-hot encoded, $\sum_{j=1}^k y_j = 1$, we have $\sum_{j \neq i} y_j = 1 - y_i$.
$$\frac{\partial L}{\partial z_i} = -y_i + y_i o_i + o_i (1 - y_i)$$
$$\frac{\partial L}{\partial z_i} = -y_i + y_i o_i + o_i - y_i o_i$$
$$\frac{\partial L}{\partial z_i} = o_i - y_i$$

Therefore, the gradient is:
$$\frac{\partial L}{\partial z_i} = o_i - y_i$$
