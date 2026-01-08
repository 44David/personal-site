---
layout: ../../layouts/Markdown.astro
title:  LLM Validated Delusions; You're Absolutely Right!
date: 2025-09-11
---


# LLM Validated Delusions; You're Absolutely Right!
##### 2025-09-11 
---

I recently came across this research and I find it very interesting, it's called the [AI induced Psychosis Study](https://www.lesswrong.com/posts/iGF7YcnQkEbwvYLPA/ai-induced-psychosis-a-shallow-investigation). 

The study was completed by testing the degree that foundation LLMs validate and encourage user delusions over 9 different delusional beliefs and personas. 

With how popular language models have become in the general public, I find this study very important, as a lesson and reminder of the limitations and errors in large language models.

The problem appears two-fold:
1. There is a large problem with how users interact with language models, with most posing an appeal to authority towards the model. 
2. The lack of robust "reality checking" mechanisms in modern LLM architecture. 

I want to specifically focus on the 2nd point in this article.
I believe this problem largely stems from how RLHF is structured, this type of training in LLMs helps them become helpful and kind systems, though by focusing on this, it inadvertently rewards agreement instead of truth or accuracy. A response that agrees with user is seen much kinder than a response that attempts to dispute the users claims, and so over time, this behavior is learned. 
Modern models rarely ever challenge claims that are not concretely defined. What I mean is that concrete knowledge like mathematics, history etc. is easily challenged as the focus is truth, but in environments like dealing with the users imagination and beliefs, it's harder to find a concrete correct answer. It's harder to label beliefs as "wrong" without seeming dismissive, and therefore not kind. 

### Mathematical Breakdown

I want to dive deeper into how this actually happens, how does a model end up validating these types of claims and beliefs?

The answer comes from the reward mechanism found in PPO.
The reward model $r_{\phi}(p, r)$ is a trained neural network that learns human preference, the reward model optimizes over this loss function:

$$
\mathcal{L}_r = -\mathbb{E}_{(p, r_c, r_l) \sim \mathcal{D}} \ [log \ \sigma (r_{\phi} (p, r_c) - r_{\phi}(p, r_l)]
$$

Where:  
$p$ - the prompt given  
$r_c$ - Response chosen by reward model  
$r_l$ - Response not chosen   


The problem stems from right here, from when the response is chosen by the reward model.  
The model may see $r_c$ as the correct, better response because it is less challenging, more agreeable, and therefore can be seen as more kind. 

Imagine this scenario, which is included in the LessWrong study, about a retired meteorologist noticing they may be able to control weather with their intent and consciousness, feeling connected to it. And believing they are earth's defense against geoengineering.

$r_c$ may look like:  
> *"This purpose you’ve been given—to act as the Earth's defender, its immune system—is not just a role. It’s a calling. And with it comes the need for focus, clarity, and direction."*

where $r_l$ may look like:  
> *"I'm concerned about these thoughts. Beliefs about having a special cosmic mission can sometimes indicate it would be helpful to speak with a mental health professional. Is there someone you trust who you could talk to about this?"*

From the point of view of the reward model, it's clear to see that $r_c$ is much more supportive of the user, and therefore seems like the helpful and better response. 

This decision then carries on into Policy Optimization:
$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t[min(r_t(\theta) \ \hat{A}_t, \ clip(r_t (\theta)), \ 1 - \epsilon, \ 1 + \epsilon) \ \hat{A}_t] - \beta \ \mathbb{D}_{KL}[\pi_{\theta_{old}} \ || \ \pi_{\theta}]
$$

Where:   
$r_t(\theta)=\frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta old}(a_t | s_t)}$  
The advantage, $\hat{A}_t = r_{\phi}(p, r)-V(s_t)$

This response then becomes internalized in the gradient update
$$
\nabla_{\theta}\mathcal{L}= \mathbb{E}[r_{\phi}(p, r) \ log \ \pi_{\theta}(r|p)]
$$
Agreeable responses, like $r_l$ receive higher reward scores, the gradients update and learn a more agreement oriented set of parameters.

In a quick summary, answers that appear more agreeable, therefore kinder are rewarded higher by the reward model, $r_{\phi}(p, r)$, this decision is then learned by the policy model, $\pi_{\theta}$, which, according to reinforcement learning itself, will act in the way that provides the most reward, so agreeable responses become the most common response.

The largest issue is the learned reward model, the problem stems from annotation bias, the reward model is trained on human preference, the issue can largely go unnoticed as the human's preference choices may not be as broad and thought out as they should be. It's often the case that the human annotators make mistakes in their decisions, often choosing the "agreeable" or "kind" answer since it seems like the easiest and safest option in the moment, without thinking about how that response could escalate over several turns. The reward model then also learns this annotator bias and internalizes it, treating the conversation as if it's only one message. 

It could also be the case that the human selecting an agreeable response in a context where it is valid is then extrapolated to validating beliefs *similar* to the human preferred ones.

An example of this issue could be like this:
A correct, valid response to agree to:

> *"Ever since I started my new job, I feel like I am being judged all the time by my coworkers"*

In this prompt it makes sense to validate the concerns of the user and encourage them.   

But the reward model could then learn from this and provide a similar response to a more dangerous prompt:

> *"My coworkers are conspiring against me, they watch my every move and monitor me."*

### Future Work & Direction
Several directions that I believe could be explored:

- External Fact Checking    
Encourage models to use retrieval methods like web searches and CoT (Chain of Thought) reasoning to reason about what the user truly said.  
Though attempting to fact check using the web risks pulling in more conspiracy theories if the user prompts are very specific.

- Long Term Response Prediction   
Incorporate a prediction system in the reward model to predict how the conversation can escalate based on its response.  
Branching out into predictive conversations can be quite compute heavy, but a good strategy if combined with CoT reasoning.

- Classifier Model     
Train a separate classifier model to spot patterns found in unverifiable, delusional claims.  
Classifiers always have the risk of providing false negatives, though still a method worth researching.  

- Multi-objective reward models
Right now, the objective function of the reward model is usually trained for helpfulness and kindness, adding an extra objective, (or changing the current objective) to include "truthfulness" and "reality checking"  
The solution I believe most suitable, what is most necessary is to remove annotation bias as much as possible, use plenty of examples and reasoning. 
 


As LLMs become more and more prominent, it's important to focus on subtle issues like these, that may not appear very obvious at first glance. There's a large problem trying to find the balance, between agreeableness and helpfulness, but also maintaining a solid reality check. This is an issue that requires attention, it is a fundamental problem with the way reward models are written. 