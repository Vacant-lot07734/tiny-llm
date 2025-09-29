# tiny-llm - LLM Serving in a Week

[![CI (main)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml/badge.svg)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml)

A course on LLM serving using MLX for system engineers. The codebase
is solely (almost!) based on MLX array/matrix APIs without any high-level neural network APIs, so that we
can build the model serving infrastructure from scratch and dig into the optimizations.

The goal is to learn the techniques behind efficiently serving a large language model (e.g., Qwen2 models).

In week 1, you will implement the necessary components in Python (only Python!) to use the Qwen2 model to generate responses (e.g., attention, RoPE, etc). In week 2, you will implement the inference system which is similar to but a much simpler version of vLLM (e.g., KV cache, continuous batching, flash attention, etc). In week 3, we will cover more advanced topics and how the model interacts with the outside world.

Why MLX: nowadays it's easier to get a macOS-based local development environment than setting up an NVIDIA GPU.

Why Qwen2: this was the first LLM I've interacted with -- it's the go-to example in the vllm documentation. I spent some time looking at the vllm source code and built some knowledge around it.

## Book

The tiny-llm book is available at [https://skyzh.github.io/tiny-llm/](https://skyzh.github.io/tiny-llm/). You can follow the guide and start building.

## Community

You may join skyzh's Discord server and study with the tiny-llm community.

[![Join skyzh's Discord Server](book/src/discord-badge.svg)](https://skyzh.dev/join/discord)

## Roadmap

Week 1 is complete. Week 2 is in progress.

| Week + Chapter | Topic                                                       | Code | Test | Doc |
| -------------- | ----------------------------------------------------------- | ---- | ---- | --- |
| 1.1            | Attention                                                   | ✅    | ✅   | ✅  |
| 1.2            | RoPE                                                        | ✅    | ✅   | ✅  |
| 1.3            | Grouped Query Attention                                     | ✅    | ✅   | ✅  |
| 1.4            | RMSNorm and MLP                                             | ✅    | ✅   | ✅  |
| 1.5            | Load the Model                                              | ✅    | ✅   | ✅  |
| 1.6            | Generate Responses (aka Decoding)                           | ✅    | ✅   | ✅  |
| 1.7            | Sampling                                                    | ✅    | ✅   | ✅  |
| 2.1            | Key-Value Cache                                             | ✅    | ✅   | ✅  |
| 2.2            | Quantized Matmul and Linear - CPU                           | ✅    | ✅   | 🚧  |
| 2.3            | Quantized Matmul and Linear - GPU                           | ✅    | ✅   | 🚧  |
| 2.4            | Flash Attention 2 - CPU                                     | ✅    | ✅   | 🚧  |
| 2.5            | Flash Attention 2 - GPU                                     | ✅    | ✅   | 🚧  |
| 2.6            | Continuous Batching                                         | ✅    | ✅   | ✅  |
| 2.7            | Chunked Prefill                                             | ✅    | ✅   | ✅  |
| 3.1            | Paged Attention - Part 1                                    | 🚧    | 🚧   | 🚧  |
| 3.2            | Paged Attention - Part 2                                    | 🚧    | 🚧   | 🚧  |
| 3.3            | MoE (Mixture of Experts)                                    | 🚧    | 🚧   | 🚧  |
| 3.4            | Speculative Decoding                                        | 🚧    | ✅   | 🚧  |
| 3.5            | RAG Pipeline                                                | 🚧    | 🚧   | 🚧  |
| 3.6            | AI Agent     / Tool Calling                                 | 🚧    | 🚧   | 🚧  |
| 3.7            | Long Context                                                | 🚧    | 🚧   | 🚧  |

Other topics not covered: quantized/compressed kv cache, prefix/prompt cache; sampling, fine tuning; smaller kernels (softmax, silu, etc)
好的，我们来推导一下 RoPE 编码后，两个向量的内积（点积）是如何只依赖于它们的相对位置和原始向量内容的。

**符号约定：**

*   `q_m`: 位置为 `m` 的原始 Query 向量 (或其一部分)。
*   `k_n`: 位置为 `n` 的原始 Key 向量 (或其一部分)。
*   `q'_m`: 经过 RoPE 旋转后的 Query 向量。
*   `k'_n`: 经过 RoPE 旋转后的 Key 向量。
*   我们考虑 `d` 维向量，RoPE 成对操作。为了简化推导，我们先看其中一对维度，即一个二维子空间。假设这一对维度是 `(x_0, x_1)`。
*   `θ_j`: 与维度对 `j` (这里我们用 `θ` 代表特定一对维度的旋转频率参数) 相关联的频率。
*   旋转角度分别为 `mθ` 和 `nθ`。

**二维子空间中的 RoPE 旋转：**

对于位置 `m` 的向量 `q_m = [q_{m,0}, q_{m,1}]`，旋转后的向量 `q'_m = [q'_{m,0}, q'_{m,1}]` 为：
`q'_{m,0} = q_{m,0} \cos(mθ) - q_{m,1} \sin(mθ)`
`q'_{m,1} = q_{m,0} \sin(mθ) + q_{m,1} \cos(mθ)`

同样，对于位置 `n` 的向量 `k_n = [k_{n,0}, k_{n,1}]`，旋转后的向量 `k'_n = [k'_{n,0}, k'_{n,1}]` 为：
`k'_{n,0} = k_{n,0} \cos(nθ) - k_{n,1} \sin(nθ)`
`k'_{n,1} = k_{n,0} \sin(nθ) + k_{n,1} \cos(nθ)`

**计算旋转后向量的内积：**

我们想计算 `q'_m ⋅ k'_n = q'_{m,0} k'_{n,0} + q'_{m,1} k'_{n,1}`。

代入上面的表达式：
`q'_{m,0} k'_{n,0} = (q_{m,0} \cos(mθ) - q_{m,1} \sin(mθ)) (k_{n,0} \cos(nθ) - k_{n,1} \sin(nθ))`
`= q_{m,0} k_{n,0} \cos(mθ)\cos(nθ) - q_{m,0} k_{n,1} \cos(mθ)\sin(nθ) - q_{m,1} k_{n,0} \sin(mθ)\cos(nθ) + q_{m,1} k_{n,1} \sin(mθ)\sin(nθ)`  (1)

`q'_{m,1} k'_{n,1} = (q_{m,0} \sin(mθ) + q_{m,1} \cos(mθ)) (k_{n,0} \sin(nθ) + k_{n,1} \cos(nθ))`
`= q_{m,0} k_{n,0} \sin(mθ)\sin(nθ) + q_{m,0} k_{n,1} \sin(mθ)\cos(nθ) + q_{m,1} k_{n,0} \cos(mθ)\sin(nθ) + q_{m,1} k_{n,1} \cos(mθ)\cos(nθ)`  (2)

现在将 (1) 和 (2) 相加：
`q'_m ⋅ k'_n =`
`  q_{m,0} k_{n,0} (\cos(mθ)\cos(nθ) + \sin(mθ)\sin(nθ))`
`+ q_{m,1} k_{n,1} (\sin(mθ)\sin(nθ) + \cos(mθ)\cos(nθ))`
`+ q_{m,0} k_{n,1} (-\cos(mθ)\sin(nθ) + \sin(mθ)\cos(nθ))`
`+ q_{m,1} k_{n,0} (-\sin(mθ)\cos(nθ) + \cos(mθ)\sin(nθ))`

**使用三角恒等式：**
*   `cos(A - B) = cosA cosB + sinA sinB`
*   `sin(A - B) = sinA cosB - cosA sinB`

应用这些恒等式：
*   `\cos(mθ)\cos(nθ) + \sin(mθ)\sin(nθ) = \cos(mθ - nθ) = \cos((m-n)θ)`
*   `\sin(mθ)\cos(nθ) - \cos(mθ)\sin(nθ) = \sin(mθ - nθ) = \sin((m-n)θ)`

代入到上面的和式中：
`q'_m ⋅ k'_n =`
`  q_{m,0} k_{n,0} \cos((m-n)θ)`
`+ q_{m,1} k_{n,1} \cos((m-n)θ)`
`+ q_{m,0} k_{n,1} \sin((m-n)θ)`
`+ q_{m,1} k_{n,0} (-\sin((m-n)θ))`  （注意这里的符号变化，因为原始是 `-\cos(A)\sin(B) + \sin(A)\cos(B)`）

整理一下：
`q'_m ⋅ k'_n = (q_{m,0} k_{n,0} + q_{m,1} k_{n,1}) \cos((m-n)θ) + (q_{m,0} k_{n,1} - q_{m,1} k_{n,0}) \sin((m-n)θ)`

**解读结果：**

1.  `(q_{m,0} k_{n,0} + q_{m,1} k_{n,1})`: 这是原始向量 `q_m` 和 `k_n` 在这个二维子空间中的内积。
2.  `(q_{m,0} k_{n,1} - q_{m,1} k_{n,0})`: 这可以看作是 `q_m` 和 `k_n` 的二维“叉积”的 Z 分量（或者与复数乘法的虚部相关）。
3.  `\cos((m-n)θ)` 和 `\sin((m-n)θ)`: 这些项只依赖于**相对位置 `m-n`** 和频率 `θ`。

**结论：** 经过 RoPE 旋转后，两个向量在特定二维子空间中的内积，是原始向量内积和叉积项与一个仅依赖于相对位置 `m-n` 的旋转因子的组合。

**推广到更高维度：**

一个 `d` 维的 Query (或 Key) 向量可以看作是由 `d/2` 个这样的二维子空间组成的。总的内积 `Q'_m ⋅ K'_n` 是所有这些二维子空间内积的和：

`Q'_m ⋅ K'_n = Σ_{j=0}^{d/2 - 1} [ (q_{m,2j} k_{n,2j} + q_{m,2j+1} k_{n,2j+1}) \cos((m-n)θ_j) + (q_{m,2j} k_{n,2j+1} - q_{m,2j+1} k_{n,2j}) \sin((m-n)θ_j) ]`

其中 `θ_j = \text{base}^{-2j/d}` 是与第 `j` 个维度对相关的频率。

**关键洞察：**

*   **绝对位置的消失：** 原始的绝对位置 `m` 和 `n` 在最终的内积表达式中只以相对位置 `m-n` 的形式出现。
*   **依赖于原始内容：** 内积结果仍然依赖于原始向量 `q` 和 `k` 的分量 (`q_{m,2j}`, `k_{n,2j}` 等)。
*   **相对位置编码：** 这证明了 RoPE 确实通过对绝对位置的旋转操作，实现了在注意力计算（内积是核心）中对相对位置的编码。

**使用复数表示（更简洁的推导）：**

我们可以将每对维度 `(x_0, x_1)` 表示为复数 `z = x_0 + i x_1`。
RoPE 旋转 `mθ` 可以表示为乘以 `e^{imθ}`：
`q'_m = q_m e^{imθ_j}` (对每个维度对 `j`)
`k'_n = k_n e^{inθ_j}`

我们想计算 `Re(q'_m (k'_n)^*)`，其中 `(k'_n)^*` 是 `k'_n` 的共轭复数。
（注意：在注意力中，我们通常计算 `Q K^T`。如果 Q 和 K 的每个维度对都看作复数，那么 `q_m k_n^T` 对应于 `Σ Re(q_{m,j} (k_{n,j})^*)`，其中 `q_{m,j}` 和 `k_{n,j}` 是未旋转的复数。旋转后，我们计算 `Re(q'_{m,j} (k'_{n,j})^*)`）。

`q'_{m,j} (k'_{n,j})^* = (q_{m,j} e^{imθ_j}) (k_{n,j} e^{inθ_j})^*`
`= q_{m,j} e^{imθ_j} (k_{n,j})^* e^{-inθ_j}`
`= q_{m,j} (k_{n,j})^* e^{i(m-n)θ_j}`

现在取其实部：
`Re(q_{m,j} (k_{n,j})^* e^{i(m-n)θ_j})`
令 `q_{m,j} (k_{n,j})^* = A + iB` （其中 `A = q_{m,0}k_{n,0} + q_{m,1}k_{n,1}`，`B = q_{m,1}k_{n,0} - q_{m,0}k_{n,1}`）
令 `e^{i(m-n)θ_j} = \cos((m-n)θ_j) + i\sin((m-n)θ_j)`

则 `(A + iB) (\cos((m-n)θ_j) + i\sin((m-n)θ_j))`
`= A \cos((m-n)θ_j) - B \sin((m-n)θ_j) + i (A \sin((m-n)θ_j) + B \cos((m-n)θ_j))`

其实部为：
`A \cos((m-n)θ_j) - B \sin((m-n)θ_j)`
`= (q_{m,0}k_{n,0} + q_{m,1}k_{n,1}) \cos((m-n)θ_j) - (q_{m,1}k_{n,0} - q_{m,0}k_{n,1}) \sin((m-n)θ_j)`
`= (q_{m,0}k_{n,0} + q_{m,1}k_{n,1}) \cos((m-n)θ_j) + (q_{m,0}k_{n,1} - q_{m,1}k_{n,0}) \sin((m-n)θ_j)`

这与我们之前用实数代数推导出的结果一致。复数表示法通常更紧凑和优雅。

这个推导清晰地显示了 RoPE 如何将绝对位置信息（通过旋转角度 `mθ` 和 `nθ`）转化为查询和键向量内积中仅依赖于相对位置 `m-n` 的调制因子。