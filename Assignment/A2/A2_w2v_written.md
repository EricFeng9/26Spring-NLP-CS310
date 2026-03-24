# CS310 Natural Language Processing
## Assignment 2 (written)
**Total points: 10**

Consider the learning objective of word2vec with SkipGram architecture: to predict a probability of an outside word $o$ given a specific center word $c$, and the probability is modeled by a naïve softmax function:

$$P(o|c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{w \in \text{Vocab}} \exp(\mathbf{u}_w^\top \mathbf{v}_c)}$$

Here $\mathbf{u}_o, \mathbf{v}_c \in \mathbb{R}^d$ are the word vectors/embeddings to be learned, and "Vocab" indicates the entire vocabulary. The parameters for all words in the vocabulary are stored in two matrices, $\mathbf{U}, \mathbf{V} \in \mathbb{R}^{d \times |\text{Vocab}|}$. For example, the $o$-th column in $\mathbf{U}$ is the outside vector $\mathbf{u}_o$, and the $c$-th column in $\mathbf{V}$ is the center vector $\mathbf{v}_c$.

### 1) 
Suppose for a particular center word $c$ and outside word $o$, let $\mathbf{y}$ be the true distribution and $\mathbf{\hat{y}}$ be the predicted distribution. Both $\mathbf{y}$ and $\mathbf{\hat{y}}$ are vectors with length equal to the vocabulary size. For $\mathbf{y}$, the $o$-th entry is $y_o = 1$ and all other entries $y_w = 0$ ($w \neq o$); for $\mathbf{\hat{y}}$, the $o$-th entry $\hat{y}_o$ is a scalar representing the probability $P(o|c)$, and $\hat{y}_w$ at $w$-th entries ($w \neq o$) are probabilities for other outside words.

Show that the cross-entropy loss $J(\mathbf{y}, \mathbf{\hat{y}}) = -\log(\hat{y}_o)$. You may describe your answer in words.

#### Answer

The cross-entropy loss is  $J(\mathbf{y}, \mathbf{\hat{y}}) = - \sum_{w \in \text{Vocab}} y_w \log(\hat{y}_w)$. 

Since $\mathbf{y}$ is a one-hot vec , $y_o = 1$ and $y_w = 0$ for all $w \neq o$, 

simplifies where $w=o$:

$$J(\mathbf{y}, \mathbf{\hat{y}}) = - (1 \cdot \log(\hat{y}_o) + \sum_{w \neq o} 0 \cdot \log(\hat{y}_w)) = -\log(\hat{y}_o)$$

**(1 points)**


---

### 2)
The cross-entropy loss $J$ is a function of $\mathbf{v}_c$, $o$, and $\mathbf{U}$, because $\mathbf{\hat{y}}$ are computed from them, and $\mathbf{y}$ is determined by $o$.

Compute the partial derivative $\frac{\partial J}{\partial \mathbf{v}_c}$. Please write the final answer in terms of $\mathbf{y}, \mathbf{\hat{y}},$ and $\mathbf{U}$. Do not use specific elements like $\mathbf{u}_o$ or $\hat{y}_w$.
#### Answer

expand the loss funct:
$$J = -\log\left(\frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{w} \exp(\mathbf{u}_w^\top \mathbf{v}_c)}\right) = - \mathbf{u}_o^\top \mathbf{v}_c + \log \sum_{w} \exp(\mathbf{u}_w^\top \mathbf{v}_c)$$
partial derivative w.r.t. $\mathbf{v}_c$:
$$\frac{\partial J}{\partial \mathbf{v}_c} = -\mathbf{u}_o + \sum_{w} \frac{\exp(\mathbf{u}_w^\top \mathbf{v}_c)}{\sum_{x} \exp(\mathbf{u}_x^\top \mathbf{v}_c)} \mathbf{u}_w = -\mathbf{u}_o + \sum_{w} \hat{y}_w \mathbf{u}_w$$
$\mathbf{u}_o = \mathbf{U}\mathbf{y}$ and $\sum_{w} \hat{y}_w \mathbf{u}_w = \mathbf{U}\mathbf{\hat{y}}$. Thus:
$$\frac{\partial J}{\partial \mathbf{v}_c} = \mathbf{U}(\mathbf{\hat{y}} - \mathbf{y})$$

**(5 points)**

---

### 3)
Compute the partial derivatives of $J$ with respect to each of the outside word vectors $\mathbf{u}_w$. There will be two cases: $w = o$ for the true outside word; $w \neq o$ for other words. Please write the final answer in terms of $\mathbf{y}, \mathbf{\hat{y}},$ and $\mathbf{v}_c$, and you can use subscript (e.g., $y_w$).

#### Answer

Using the expanded form of $J$, consider two cases for $\mathbf{u}_w$:
1. For  $w = o$:
  $$\frac{\partial J}{\partial \mathbf{u}_o} = -\mathbf{v}_c + \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{x} \exp(\mathbf{u}_x^\top \mathbf{v}_c)} \mathbf{v}_c = (\hat{y}_o - 1)\mathbf{v}_c$$

2. For other words $w \neq o$:
  $$\frac{\partial J}{\partial \mathbf{u}_w} = 0 + \frac{\exp(\mathbf{u}_w^\top \mathbf{v}_c)}{\sum_{x} \exp(\mathbf{u}_x^\top \mathbf{v}_c)} \mathbf{v}_c = \hat{y}_w \mathbf{v}_c$$

  

  Since $y_o = 1$ and $y_w = 0$ for $w \neq o$, both cases can be unified as:
  $$\frac{\partial J}{\partial \mathbf{u}_w} = (\hat{y}_w - y_w)\mathbf{v}_c$$

**(4 points)**
