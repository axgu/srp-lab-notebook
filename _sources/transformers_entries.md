# Transformers

**Transformers**

*08/10/2022, 8:40 am*


Synopsis: I will learn about self attention and the transformer architecture.

Data: I read material about the self attention mechanism and how it fits into transformers.

Files:
* [Recurrent Neural Network Notes](./recurrent_nn.md)

*5:10 pm, 510 minutes*

---

**Simple Transformer Implementation**

*08/11/2022, 8:55 am*


Synopsis: I will implement a basic transformer architecture.

Data: I built a transformer model for movie clip classification based on the TransformerEncoder layer in Pytorch. I tested the model using a single head for attention and a single layer of the transformer block.

Files:
* [Single-head Transformer](./hcp_data/transformer.ipynb)

*3:28 pm, 393 minutes*

---

**Transformers with Multi-head Attention**

*08/12/2022, 8:35 am*


Synopsis: I will test hyperparamters of the transformer architecture.

Data: I built a transformer model similar to the initial one created yesterday. I trained the model at a lower learning rate and smaller hidden dimensions. I also tried adding a multi-head attention mechanism and multiple layers of the transformer block, resulting in a higher validation accuracy.

Files:
* [Multi-head Transformer](./hcp_data/transformer_multihead.ipynb)

*3:25 pm, 410 minutes*

---

**Transformers Training and Evaluation**

*08/15/2022, 8:45 am*


Synopsis: I will add position encoding to my transformer architecture.

Data: I met with Ms. Bosse and Dr. Pessoa to discuss the progress of my project and next steps. I also tried adding a position encoding function to the transformer models previously implemented. The purpose of position encoding is to differentiate the time steps (i.e. keep sequential order).

Files:
* [Position Encoding Transformers](./hcp_data/transformer_position.ipynb)

*4:55 pm, 490 minutes*

---

**Transformers Summary**

*08/16/2022, 8:40 am*


Synopsis: I will summarize the work done with transformers.

Data: I retrained the multi-head transformer with position encoding using more epochs but a lower learning rate to the model would converge. I also wrote a summary of what I have done with transformers, including information about data processing, transformer encoder architecture, self-attention mechanism, training, and evaluation.

Files:
* [Transformers Summary](./transformers_summary.md)

*5:10 pm, 510 minutes*

---
