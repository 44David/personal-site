---
layout: ../../layouts/Markdown.astro
title: Inducing reasoning in small language models
date: 2026-01-13
---


# Inducing reasoning in small language models

##### 2026-01-13
---

Small models rarely reason well, especially when it comes to verifiable mathematical reasoning tasks, which is why I decided to train qwen-0.5.b-reasoning, a SFT version of Qwen2.5-0.5B, 
Using SFT and SCoTD  lead to great improvements in reasoning capabilities in a relatively small parameter model. 

You can download both versions of my model here:

[https://huggingface.co/44David/qwen-0.5b-reasoning-v1](https://huggingface.co/44David/qwen-0.5b-reasoning-v1) <br>
[https://huggingface.co/44David/qwen-0.5.b-reasoning-v2](https://huggingface.co/44David/qwen-0.5b-reasoning-v2)


The setup I used was heavily inspired by the Symbolic Chain-of-Thought Distillation paper https://arxiv.org/abs/2306.14050.

In the original paper the training process is as follows:
- Have dataset $\mathcal{D}$
- Teacher model (significantly larger than target model) generates 6-10 chain-of-thought traces for each entry in $\mathcal{D}$ 
- SFT on target model randomizing which of the $n$ reasoning traces to use

The reason for generating so many traces is for a uniqueness and variance between each trace, making sure the target model doesn't overfit on one path of reasoning.

for my model, I decided to generate only 1 reasoning trace for each question, but I ensured they were of high quality and used a teacher model with a specialty in mathematical reasoning, which in this case was Deepseek-math-7B-instruct.

# Version 1 Results
My first attempt at training qwen-0.5.b-reasoning ended up with very interesting results, mainly due to my training environment.

For the version 1, I decided that my target dataset, $\mathcal{D}$ would be [gsm8k](https://huggingface.co/datasets/openai/gsm8k), a dataset containing over 8,000 entries of grade school math problems. I decided this would be a good starting point as Qwen2.5-0.5B, my base target model already struggled greatly with these problems. 

And so I created my reasoning trace dataset with my teacher model, you can see that dataset [here](https://huggingface.co/datasets/44David/gsm8k-reasoning-traces).

Here also is an image of the training, you can look more closer at the details of both v1 and v2 here on my wandb report:
[https://api.wandb.ai/links/davidsoane/1ta1f5s7](https://wandb.ai/davidsoane/qwen-reasoning-0.5B/reports/qwen-0-5b-reasoning-graphs--VmlldzoxNTY0MTUyNA)


<img src="/images/v1-stats.png" alt="v1 stats" width="700" />

Here are some example problems with v1:
```
Prompt: A bakery makes 12 batches of cookies. Each batch has 24 cookies. They sell 3/4 of all the cookies. How many cookies are left?


qwen-0.5b-reasoning-v1: <think> The bakery makes 12 batches of cookies, and each batch has 24 cookies. So, the total number of cookies made is 12 * 24 = 288 cookies. They sell 3/4 of all the cookies. To find out how many cookies they sell, we multiply the total number of cookies by 3/4. So, the number of cookies sold is 288 * 3/4 = 288 * 0.85 = 256 cookies. To find out how many cookies are left, we subtract the number of cookies sold from the total number of cookies made. So, the number of cookies left is 288 - 256 = 32 cookies. So the answer is $\boxed{32}$.</think> The bakery has made 12 x 24 = 288 cookies. They sold 288 x 3/4 = 256 cookies. So, 288 - 256 = 32 cookies are left.

```

As opposed to Qwen2.5-0.5B, the base model:
```
Prompt: A bakery makes 12 batches of cookies. Each batch has 24 cookies. They sell 3/4 of all the cookies. How many cookies are left?

Qwen2.5-0.5B: There are 36 cookies left after selling 3/4 of the cookies.
```


We can see a clear beneficial improvement here between my model and the base model, with qwen-0.5b-reasoning, it generates clean ```<think>``` tags to separate its reasoning and answer, it also has its CoT answer in ```\\boxed{}```, which makes it much easier for the user, but more specifically for reinforcement learning, where tasks like math questions, where the answers must be easily identifiable in order to verify correctness.
Which, when compared to the base model Qwen which tries to just spit out an answer.

Though something you may have noticed from the both model outputs, is that they are both wrong! The correct answer for this question is 72 cookies remaining.

We can closer inspect the error that happened in the reasoning trace, where the model believed 3/4 to be equal to 0.85, and 288 * 0.85 to be 256, when it's actually ~245. 

A possible reason for the errors in arithmetic could be due to how language models process information, which in chunks, called tokens:

## Tokenization in Language Models

You may have heard of the famous prompt of "How many r's are in strawberry?"
The reason early language models struggled with this questions is again because of tokens.
The tokenization system in language models reads input in chunks, leaving some words whole, and others as 2, or 3 parts of tokens, an example of tokenization can be found in this image:

<img src="/images/subword-tokenization.png" alt="subword tokenization" width="700" />

Notice how words like LLM are made up of the purple part, "LL" and also the green, being only the "M".
The reason we don't use word-level tokenization is because it would require a huge vocabulary (unique token for each word), and the fact that we cannot add any new words at test-time. As opposed to the system you see above, which is called a subword tokenization, where tokens are mostly parts of words, for example, there might be a token ```cious``` because it is common in many words: deli-cious, pre-cious, cons-cious.

And so back to the question, when the model reads strawberry, it may not see all the letter r's separately, but as a mix of tokens, *maybe -erry is a token here.*

And the same problem with subword tokenization may be occurring when we try to do arithmetic, seeing the addition problem as a mix of tokens, not the numbers themselves. 

*Note that this is speculative, the real issue is still not well known, another hypothesis is that transformers inherently are not good at symbolic manipulation* 

Though, with all things considered, this is a pretty good start, it has a good format and properly thinks through problems step by step, and after coming up with an answer in it's thinking, it formats the reasoning in short for an official answer to the user. All of this is a massive improvement over the base capabilities of Qwen2.5-0.5B.

Though a few other tests showed a couple weaknesses with version 1, conversational tasks ended in a complete failure due to catastrophic forgetting:

```
Prompt: Hello!

qwen-0.5b-reasoning-v1: <think> Let's denote the number of ducks the farmer sold as D. The farmer sold 5 chickens, so the total number of animals he sold is D + 5. The price of each chicken is $8, so the total earnings from selling chickens is 5 * $8 = $40. The price of each duck is $10, so the total earnings from selling ducks is D * $10 = $10D. Therefore, the total earnings from selling all the animals is $40 + $10D. The farmer spends half of his earnings on a new wheelbarrow, so the cost of the wheelbarrow is 0.5 * ($40 + $10D) = $20 + $5D. Someone sees the wheelbarrow and agrees to pay the farmer double what the farmer paid, so the farmer earns 2 * ($20 + $5D) = $40 + $10D. We know that the farmer earns another $60 from selling the wheelbarrow, so $40 + $10D = $60. Subtracting $40 from both sides of the equation gives $10D = $20. Dividing both sides of the equation by $10 gives D = 2. So, the farmer sold $\boxed{2}$ ducks.</think> The wheelbarrow cost $30 because 60 / 2 = 30 The farmer earned $60 from selling fowl because 30 / (1/2) = 60 The farmer earned $40 selling chickens because 5 x 8 = 40 The farmer earned $20 selling ducks because 60 - 40 = 20 The farmer sold 2 ducks because 20 / 10 = 2
```

```
Prompt: What is the capital of France?

qwen-0.5b-reasoning-v1: <think> The capital of France is France. The answer is $\boxed{ France}$.</think> The capital of France is: France = 146ks The capital of Germany is: Germany = 98ks The capital of Russia is: Russia = 31ks The capital of Belgium is: Belgium = 21ks The capital of Canada is: Canada = 11ks The capital of Spain is: Spain = 18ks The capital of Italy is: Italy = 8ks The capital of France is: 146ks
```

it seems that version 1 has fully become a reasoning model, "forgetting" all of the other objectives and tasks it was previously trained on. This could be due to an aggressive learning rate which moved the model too far from its original point in the optimization  landscape.

# Version 2 Results 

With the failures in version 1, I decided to keep the same reasoning dataset I created, as it seemed to have a great improvement on reasoning capabilities. 

What I did instead was add another dataset into the training loop, a dataset that was solely focused on assistant and conversational tasks, for this I chose the [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset.
I did a 70-30 split by interleaving both datasets, with 70% of the dataset now being my reasoning dataset, and the 30 being alpaca.

<img src="/images/v2-stats.png" alt="v2 stats" width="700" />

This mixed dataset approach greatly improved performance in both sets of tasks, seeing improvements in arithmetic with reasoning questions and more natural conversation :

```
Prompt: A bakery makes 12 batches of cookies. Each batch has 24 cookies. They sell 3/4 of all the cookies. How many cookies are left?

qwen-0.5b-reasoning-v2:   
<think> The bakery makes 12 batches of cookies, and each batch has 24 cookies, so there are a total of 12 * 24 = 288 cookies. They sell 3/4 of all the cookies, so they sell 3/4 * 288 = 216 cookies. Therefore, there are 288 - 216 = 72 cookies left. The answer is $\boxed{72}$.</think> The bakery has 12 * 24 = 288 cookies. They sell 288 * 3/4 = 216. Thus there are 288 - 216 = 72 cookies left.

```

```
Prompt: Hello!

qwen-0.5b-reasoning-v2: Hello! How can I help you?
```

```
Prompt: What is the capital of France?

qwen-0.5b-reasoning-v2: The capital of France is Paris.
```

The improvements are night and day, previously with version 1, besides the obvious flaw of hallucinating a whole math problem when the prompt was simply "Hello!",  the model would include think tags for prompts that did not necessarily need them, with version 2, we can see that the model only reasons when it seems necessary, I noticed to be quite interesting and will be digging deeper into why this happens with my next article, where I perform mechanistic interpretability on both models to see failure points, attention patterns and what exactly could cause this to happen.  

The model generates multi-step reasoning chains but makes algebraic errors:
```
Prompt: John is 3 times as old as his sister. In 5 years, he will be twice as old as his sister. How old is John now?

qwen-0.5b-reasoning-v2: <think> Let's denote John's current age as J and his sister's current age as S. From the problem, we know two things: 1. John is 3 times as old as his sister. This can be written as: J = 3S. 2. In 5 years, John will be twice as old as his sister. This can be written as: J + 5 = 2 * (S + 5). Now, we have a system of two equations, and we can solve it step by step. First, let's substitute the first equation (J = 3S) into the second equation: 3S + 5 = 2S + 10. Then, let's subtract 3S from both sides of the equation: 5 = 2S + 10 - 3S. Simplify the right side of the equation: 5 = 7S. Finally, let's divide both sides of the equation by 7 to solve for S: S = 5 / 7. Since S represents John's sister's age, we can find John's age by substituting S = 5 / 7 into the first equation: J = 3 * (5 / 7) = 15 / 7 = 2. So, John is currently 2 years old. So the answer is $\boxed{2}$.</think> Let X be John’s age now. So his sister is X/3 years old now. In 5 years, John will be X + 5 years old. In 5 years, his sister will be (X/3) + 5 years old. So (X + 5) = 2 * (X/3) + 10. So (X + 5) = (2X/3) + 10. So (X + 5) = (4X/3) + 10. So (X + 5) = 4X + 10 - 10. So (X + 5) = 4X. So X = 5 / 4. So John is 5 / 4 * X = 5 / 4 * 4 = 5 years old.
```


Something interesting also is that the model appears to come up with 2 separate answers, in its internal reasoning, it ends up with the answer of 2, but with its final user response it ends up with 5. Both answers are incorrect, as the correct answer here is 15.

It seems that the model has learned the format of reasoning, but not a reliable system of arithmetic.

Another place the model struggled was creative writing, for example, when asked to tell a joke:
```
Prompt: Tell me a joke

qwen-0.5b-reasoning-v2: What did the egg say when it hit the wall? It was hit, not broken.
```

Though this might just be me and qwen having different tastes in jokes, it doesn't appear to make much sense, and it has that basic setup of question then punchline. 


# Key Findings & Conclusion

This was a very interesting project, and there is still much I want to analyze in these two models, especially v2 detecting "math mode" and "conversational mode" with think tags.

But I want to outline what I found interesting and what I discovered in this work:
- Small models can in fact learn conditional reasoning using the SCoTD format of using a teacher model
- Mixed training methods (like interleaving conversational and reasoning datasets) seems to have *improved* arithmetic ability - *will have to test this further, as this is a small sample size*
- catastrophic forgetting is very much a risk when working with very narrow and specific datasets
- ```<think>``` tags seem to help induce reasoning in language models 
- 0.5B parameter models are great for research, though not necessarily viable for production due to common flaws like hallucinations. 
