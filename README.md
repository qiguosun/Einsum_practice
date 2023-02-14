# Einsum_practice
The detailed describ of Einsum can ben found here(https://rockt.github.io/2018/04/30/einsum).

Einsum is an elegant tensor operation implemented in numpy via np.einsum, PyTorch via torch.einsum, and Tensorflow via tf.einsum.  Suppose we have ${\color{green}c_j} = \sum_i\sum_k {\color{red}A_{ik}}{\color{blue}B_{kj}}$, it can be written as the equation string ${\color{red}ik},{\color{blue}kj}$

Let a tensor ${\color{red}\mathcal{T}}\in\mathbb{R}^{N\,\times\,T\times\,K}$,  a projection matrix ${\color{blue}\mathbf{W}}\in\mathbb{R}^{K\,\times\,Q}$, a target tensor with ${\color{green}C_{ntq}}$.
Then using einsum expression to calculate the target,

${\color{green}C_{ntq}} = \sum_k {\color{red}T_{ntk}}{\color{blue}W_{kq}} = {\color{red}T_{ntk}}{\color{blue}W_{kq}}.$

Overall, a typical call of Einsum can be expressed as follows,

${\color{green}\textbf{result}} = \text{einsum}("{\color{red}\square\square},{\color{purple}\square\square\square},{\color{blue}\square\square}\,\text{->}\,{\color{green}\square\square}", {\color{red}\text{arg1}}, {\color{purple}\text{arg2}}, {\color{blue}\text{arg3}})$

where $\square$ is a placeholder for a character identifying a tensor dimension,
the args are the input matrices or tensors.
