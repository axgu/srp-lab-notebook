# Attention Methods

## Sequence to Sequence Models
Sequence to sequence models convert a sequence of one type to a sequence of another type. The encoder decoder architecture is often used to achieve this. Both the encoder and decoder are built with RNNs, but the entire model is trained end-to-end (i.e. from input into encoder to output from decoder).

At each time $t$, recurrent neural networks are composed of the input, internal or hidden state, and output. 

The encoder inputs the data in time steps. It disregards the output at each $t$, but stores the internal state at $t$ to calculate the internal state at $t+1$. The final information is the hidden state of the last time step, called the context vector.

The context vector is passed into the decoder sequentially. The decoder calculates the next internal state using the input context vector data and the previous internal state. This information is used to calculate the output at each time step $t$.

### Problem
The issue with standard encoder decoder architecture arises when input sequences increase in length, as it becomes very difficult to capture information from the entire input sequence in one vector. Since this last internal state of the encoder is the only vector passed through to the decoder, performance of the model decreases.


## Attention
When applied to a sequence to sequence model, attention determines and focuses the model on the most relevant input features for a certain output. In general, this is done by storing all hidden states in a matrix, instead of passing only the vector representing the last hidden state to the decoder. Having access to all internal state information, it creates mappings between outputs at all time steps of the decoder and all internal states of the encoder to determine and select features most influential in producing an output at a certain time step. The "attention" given to an input feature is quantified with alignment scores.

Several attention mechanisms have been introduced.


### Bahdanau vs Luong Approach
The attention mechanisms proposed by Bahdanau et al. and Luong et al. involve three aspects: the encoder model, alignment scores, and the decoder model. The main difference in approach between the two processes is when the decoder RNN is utilized.

#### Encoder Model
The first step for both methods is to pass inputs into the encoder RNN and store all hidden states.

#### Alignment Scores
The process of calculating alignment scores differs slightly between the Bahdanau and Luong methods because the decoder RNN is used at different times.

Bahdanau et al. calculated alignment scores using the following equation: 

$$
\text{score}_{\text{alignment}} = W_{\text{combined}} \cdot \tanh(W_{\text{decoder}} * H_{\text{decoder}} ^{(t-1)} + W_{\text{encoder}} \cdot H_{\text{encoder}})
$$

$H_{\text{decoder}} ^{(t-1)}$ is the vector representing the decoder hidden state at time step $t-1$ and $H_{\text{encoder}}$ is the matrix of all encoder outputs/hidden states.

Luong et al. calculated alignment scores using three different equations:

* Dot:

$$
\text{score}_{\text{alignment}} = H_{\text{encoder}} \cdot H_{\text{decoder}}
$$

* General:

$$
\text{score}_{\text{alignment}} = W(H_{\text{encoder}} \cdot H_{\text{decoder}})
$$

* Concat:

$$
\text{score}_{\text{alignment}} = W \cdot \tanh(W_\text{combined}(H_{\text{encoder}} + H_{\text{decoder}}))
$$

$H_{\text{decoder}}$ is the vector representing the decoder hidden state at time step $t$ and $H_{\text{encoder}}$ is the matrix of all encoder outputs/hidden states.

Both methods applied the $softmax$ function on the alignment scores vector to obtain the attention weights. The context vector was then calculated by multiplying the attention weights with the encoder outputs.

#### Decoder Model
For the Bahdanau attention method, the decoder RNN was used last. At time $t$, the context vector was concatenated with the $t-1$ decoder output. This, along with the $t-1$ hidden state, was fed into the decoder RNN to calculate the time step $t$ hidden state. The new hidden state was passed through a classifier to obtain the output.

For the Luong attention method, the decoder RNN was used first, prior to calculating the alignment scores. Thus, the hidden state at time step $t$ instead of $t-1$ was used for alignment scores and producing the final output.


### Soft vs hard attention
Attention methods highlight the most relevant input features using probability distributions; the $softmax$ activation function determines a weight that corresponds with each input.

In soft attention, a weighted average of features and their weights is computed, and the result is fed into the RNN architecture. Generally, soft attention is parameterized by differentiable and thus continuous functions, so training can be completed via backpropagation.

However, in hard attention, the obtained weights are used to sample one feature to input into the RNN architecture. Hard attention methods are generally described by discrete variables. Because of this, other techniques must be used to train structures other than standard gradient descent, which depends on differentiable functions.


### Global vs local attention
Global attention refers to when attention is calculated over the entire input sequence. Local attention refers to when a subsection of the input sequence is considered.


## Resources
* [Sequence to Sequence Models](https://www.analyticsvidhya.com/blog/2020/08/a-simple-introduction-to-sequence-to-sequence-models/)
* [What is attention mechanism?](https://towardsdatascience.com/what-is-attention-mechanism-can-i-have-your-attention-please-3333637f2eac)
* [Attention by Lilian Weng](https://lilianweng.github.io/posts/2018-06-24-attention/)
* [Attention Mechanism](https://blog.floydhub.com/attention-mechanism/)
* [Neural Machine Translation by Jointly Learning to Align and Translate by Bahdanau et al.](https://arxiv.org/pdf/1409.0473.pdf)
* [Effective Approaches to Attention-based Neural Machine Translation by Luong et al.](https://arxiv.org/pdf/1508.04025.pdf)
