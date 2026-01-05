---
layout: ../../layouts/Markdown.astro
title: Mathematical Foundations of Self-Attention Mechanisms
---


# Mathematical Foundations of Self-Attention Mechanisms
##### 2025-08-15 

---
# Why Self-Attention?
First off, lets discuss why the attention mechanism has become so prevalent almost everywhere in the deep learning space.

**The main problem that self-attention solves is the long-dependency problem**.
Before the transformer and attention, the [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) was the most common for language modelling, though it ran into trouble with something called long-range context understanding. Where the beginning words of a large sentence are "forgotten" as the model processes more and more words. This happens because of the *vanishing gradient problem*.  
Mathematically, we can see this problem happen:

Given an input sentence $\mathbf{S}$, the model sequentially processes $\mathbf{S_i}$ at each step:

$$
h_t = \sigma(W_h h_{t-1} + W_s s_t + b)
$$

where  
t represents the step  
$\sigma$ represents the activation function  
$s_t$ is the single word that is processed  

You can think of $h_t$, the hidden state as being the memory of the model.  

The vanishing problem rises as we begin to backpropagate the network in order to compute the gradient: 

 $$
\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t}^{T-1} \frac{\partial h_k}{\partial h_{k-1}}
$$

It specifically lies here, in the Jacobian:

$$
\frac{\partial h_k}{\partial h_{k-1}} = W_h^T \cdot  \ diag(\sigma ` (z_k))
$$

Where $z_k$ is the pre-activation neuron computation.

If either  
$\| W_h \| < 1$   
$\sigma `(z_k) < 1$

These are the root causes of vanishing gradients,  
$W_h$ and $\sigma `$ are commonly susceptible to this because of the math which creates them.   

For $W_h$, initialization schemes like *He* and regularization techniques  like the L2 make matrix values small to prevent exploding gradients.    
And the activation functions used for $\sigma `(z_k)$ are designed to prevent the same aforementioned problem.

The fundamental problem with RNN's come from the foundation it's built on. Modifying the initialization scheme or the activation functions would then lead to other, larger problems (like exploding gradients, where gradient values are far too large) and wouldn't fix anything.



**Self-attention largely solves this by weighing each word with a score based on its importance.**

# Self Attention

An intuitive way to understand self-attention is to think about our own human attention, when you read a book, you don't memorize every single word you read, only the important parts: the plot, character names, personalities etc. 

**Self-Attention is a way to mathematically model the importance of words in a similar way humans do**

Formally, self-attention is denoted as:
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V
$$

Where:

$Q \in \mathbb{R}^{d_{model} \times d_k} = XW_q \in \mathbb{R}^{512 \times 64}$  
$K \in \mathbb{R}^{d_{model} \times d_k} = XW_k \in \mathbb{R}^{512 \times 64}$  
$V \in \mathbb{R}^{d_{model} \times d_v} = XW_v \in \mathbb{R}^{512 \times 64}$  

$X$ being the encoded embedding input sequence  
$d_k = d_v = d_{model}$
$d_{model}$ = dimension size of model (*usually 512*)

$W_{k, q, v}$ are the projection matrices that *project* the input $X$ onto the 3 spaces (Query, Key, Value). Usually initialized with He Initialization $W_{i, j} \ \sim \  \mathcal{N} \ (0, \frac{2}{n_{in}})$ (*Where $n_{in}$ is the number of input neurons*)  

The numerator of $QK^T$ essentially computes the attention score, we say that this numerator is sort of matching the Query representation for the token to the Key representation of it.  

We then scale by $\frac{1}{\sqrt{d_k}}$ to ensure values are not too large, preventing cases where only a few words are attended to (*focused on, paid attention to*).  

$softmax()$ is used to turn the attention scores into a probability distribution ensuring a row adds up to 1.  

At this point we have our attention scores for the sequence, but we must multiply it by $V$ so we can actually apply these attention scores to the words.

Self attention differs from RNN's in the sense that it is not sequential, instead of processing word by word, the entire input sequence $X$ is given to the mechanism. Self-attention finds its order in input from [positional encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)  


### *What do Q, K, V represent?*
Think of each matrix learning this:

----

$Q$ (Query) learns and understands the question of "What am I look for in this token?" Where each row of Q represents what information the current token (word) is seeking.  
  
For an example, lets say the current token is "run", The row of $Q$ containing the representation of "run" might contain an encoded representation asking "what is running? Who is running?"

-----


$K$ (Keys) learns the question "What information can this token provide?" Each row represents information about the token  
  
If we continue with the current token of "run", $K_i$ containing the token "run" might say "This word is an action, something that a thing can do"

----

$V$ (Values) says "What do information do I *actually* have?" Where each row contains the actual information the token represents.

----

## Multi-Head Attention
Think of the entire process of computing attention scores: Initializing projection matrices, creating the project inputs of Q, K, V, and then finally computing the attention scores using the attention formula; All as one head. 

And so as the name suggests, *Multi*-Head attention is the combination of multiple of these attention heads.

A multi-head attention block contains n heads. This means we initialize n different sets of projection matrices ($W_{q_n}, W_{k_n}, W_{v_n}$), and compute n unique attention scores. 

For multi-head attention, 
$d_k = d_v = d_{model} / n$

The output is still the same as the $n$ number of heads are concatenated: 
$$
MultiHeadOut = Concat(\textbf{head}_{a_1}, \textbf{head}_{a_2}, \dots, \textbf{head}_{a_n})
$$ 

We then multiply $MultiHeadOut$ by a initialized matrix of the same shape $W_o \in \mathbb{R}^{d_{model} \times d_{model}}$ for the final output.

By using multiple unique attention heads, we can compute a more accurate and better attention score.
Think about it this way:
With each head having a different set of random projection matrices, each head will learn something different about the tokens, each head will focus on a different part about that token in a unique way, when they all come together through concatenation that unique angle is added to the way the final attention is computed

# Time Complexity on larger sequences
$n$ - sequence length  
$d$ - hidden dimension size  

The time complexity for RNN's is:  
Training - $O(n \cdot d^2)$  
Inference - $O(n \cdot d^2)$  
$d^2$ because of the hidden state computation:
$$
h_t = \sigma (W_h h_{t-1} + W_x x_t + b)
$$
For 1 time step ($n$) we must compute $W_h h_{t-1}$ and since $W_h$ has shape ($d \times d$), we compute $d^2$


Time complexity of self-attention:
Training - $O(n^2 \cdot d)$  
Inference - $O(n^2 \cdot d)$  
$n^2$ is from the fact that we compute $QK^T$, which has shape ($n \times n$) given that both $Q$ and $K$ are shape ($n \times d$). We compute $n \times n$ because each token (word) needs an attention score with respect to all other tokens.  

Though the time complexities appear similar in theory, in application, the fact that RNN's are sequential, and that attention allow for *parallelization*, makes attention much faster.

Parallelization means that while an RNN takes 100 steps for a input sequence of 100 words **no matter what**, an attention mechanism can compute *all* matrix multiplication and attention scores at the exact same time thanks to parallelization.   

