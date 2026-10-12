# HW1 学习指南

四个 Notebook 的学习顺序是：**张量操作 → 自动求导和数据管线 → 从零搭网络 → 用实验检验模型设计**。
Notebook 中的实现可以作为参考；学习时建议把每个小函数遮住，自己推导和重写，再对照结果。

## 运行与查看结果

在有 `torch`、`torchvision` 的 Python 环境中，从本作业目录运行：

```bash
cd hw1_pytorch_tutorial
python test_hw1.py
python run_notebooks.py
python plot_results.py
```

`run_notebooks.py` 顺序执行真实 Notebook 单元，并把运行输出写回文件。
当前 ex4 使用便于 CPU 运行的小模型：patch 大小 7（每图 16 个 token）、特征宽度 32、2 个编码器块、
FFN 宽度 128，GLU 宽度 85。三个变体只更换 FFN，其余设置保持一致。
ex3 默认训练 3 个 epoch；ex4 默认训练 5 个 epoch，对 FFN、GEGLU、SwiGLU 各运行 3 个种子。
首次运行会下载公开 MNIST 数据。指标保存在 `results/`，可在另一个装有 `matplotlib`、`numpy` 的环境运行绘图脚本。
绘图工具不导入提交的 Notebook；脚本会把生成的图作为标准图片输出附到 Notebook 的最后一个代码单元。
当前机器用于验证的 Python 是 `/home/ykb/miniconda3/envs/lerobot/bin/python`；
绘图使用已有的 `/home/ykb/miniconda3/envs/ms3-lerobot/bin/python`，不改动这两个环境。

只运行一个部分或先检查整条流程时：

```bash
python run_notebooks.py --exercises 1 2
python run_notebooks.py --exercises 3
python run_notebooks.py --exercises 4 --ex4-epochs 1 --seeds 0
```

ex4 的执行脚本会把实际使用的 epoch/种子配置写入 Notebook，便于让输出与配置对应。
缩短训练是调试方式；结论应根据完整实验的真实结果来写。
直接用 Jupyter 也可以：从作业目录启动 `jupyter lab`，逐个打开 `src/ex*.ipynb`，从上到下运行。
四个 Notebook 相互独立，不需要先执行另一个 Notebook 才能工作。

需要新建环境时，可以沿用 README 的 Python 3.12 配置：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision jupyter matplotlib numpy
jupyter lab
```

## 每题都采用这套学习方法

1. **先写输入和输出的形状。** 同时记录 dtype、device，以及输入是否允许被修改。
2. **用很小的数手算。** 例如用 2×3 张量检查 broadcasting；用长度为 2 和 3 的序列检查 padding。
3. **写出公式或轴的对应关系，再看实现。** 理解为什么要使用某个维度，避免只背 API。
4. **与参考实现比较数值及梯度。** 基础层对比 PyTorch；掩码、空序列、随机种子要单独检查。
5. **合上参考代码重写。** 最后改变形状和参数，再解释为什么实现仍然成立。

遇到错误先打印 `shape`、`dtype`、`device`；遇到 `view` 错误再看 `stride()` 和 `is_contiguous()`。
如果只能解释代码做了什么，却不能解释少一行会发生什么，就再做一个反例。

## ex1：把形状、内存和掩码学扎实

先分清 `reshape` 与 `view`：转置改变 strides，可能不再连续；`view` 不能随意复制，`reshape` 必要时能复制。
`expand` 不复制数据，扩展后的多个位置可能共享存储；`repeat` 会复制。

练习：令 `x.shape=(2,3,4)`，沿中间维度求均值。
`x.mean(1)` 得到 `(2,4)`；`x.mean(1, keepdim=True)` 得到 `(2,1,4)`，后者可以直接与 `x` 相减。
在纸上从最右侧开始对齐维度，解释前者为什么不能按预期广播。

再理解 `einsum("bhid,bhjd->bhij", q, k)`：保留批次 b、头 h、query 位置 i、key 位置 j，只求和特征维 d。
不要背字符串；每次从输入/输出轴写出来。接着推导 `weights @ v` 对应的 einsum。

稳定 softmax 的公式是：`exp(x - max(x)) / sum(exp(x - max(x)))`。
减去最大值不改变概率，因为分子分母同时乘了同一个缩放因子，却能避免指数溢出。
全掩码行不能直接对全 `-inf` 做 softmax，否则会出现 `-inf - (-inf)`，得到 NaN。
实现先把这种行换成安全输入，最后把无效位置置零，数值和梯度都保持有限。

**需要记住：** `masked_mean` 中 True 表示保留，而 `masked_softmax` 和 padding mask 中 True 表示无效。
不能根据变量名猜含义，要先读函数契约。

掌握标准：能从零写出直方图、masked mean、因果 mask，解释为什么不需要遍历批次和 token。

## ex2：理解完整训练步骤

先手推两个梯度：

- `f(x)=sum(x²)`，梯度为 `2x`。
- `f(a,b)=sum(a²+ab)`，对 a 的梯度为 `2a+b`，对 b 为 `a`。

`autograd.grad` 返回梯度；`backward` 把梯度累加到叶张量的 `.grad`。
这里使用独立的叶张量副本，因此多次调用不会污染输入的梯度。
练习连续调用两次 `backward`，观察不清零时为什么结果翻倍。

Dataset 负责一个样本；DataLoader 负责批次；collate_fn 负责把不同长度的样本整理为相同形状。
padding mask 按真实长度生成，不能通过 `tokens == 0` 判断，因为 0 也可能是合法 token。
next-token 训练用 `tokens[:-1]` 预测 `tokens[1:]`，手写一个 `[3,5,7]` 的例子检查是否错位。

AdamW 保留每个参数的一阶矩 m、二阶矩 v 和步数 t：

```text
m ← β1 m + (1-β1) g
v ← β2 v + (1-β2) g²
m̂ = m / (1-β1^t), v̂ = v / (1-β2^t)
p ← (1-lr·weight_decay) p - lr·m̂/(sqrt(v̂)+eps)
```

练习用一个标量参数算第一步，解释初始矩为零时为什么需要偏差修正。
更新参数和优化器状态应该在 `no_grad` 内进行；衰减参数与修改梯度是不同的操作。

掌握标准：不用看代码，按顺序说出 train → zero_grad → forward → loss → backward → step，
并说明漏掉 zero_grad、backward、step 各会发生什么。

## ex3：从公式到可训练模块

`nn.Parameter` 的意义是让模块注册并暴露可训练参数，而不只是存一个 Tensor。
Linear 的权重为 `(out_features,in_features)`，计算 `x @ weight.T + bias`。
对任意前导批次维度都应成立，输入可以是 `(B,T,D)`。

LayerNorm 在最后一个维度减均值并除标准差，方差用总体方差 `unbiased=False`。
RMSNorm 不减均值，只除以 `sqrt(mean(x²)+eps)`。
练习输入一个常数向量，比较两者输出为什么不同。

Dropout 在训练时以概率 p 丢弃，保留项除以 `1-p`，这样输出期望不变；eval 时直接返回输入。
训练和测试曲线需要结合模型模式看，不能把含随机 dropout 的训练结果当作确定性评估。

MNIST 分类的形状变化是：

```text
(B,1,28,28) → flatten → (B,784) → MLP → (B,128) → head → (B,10)
```

输出 logits 不需要提前做 softmax；交叉熵为目标类别负 log 概率的平均值。
这里用 log-softmax 保证稳定，不能先 softmax 再取 log，避免小概率下溢。
训练函数保持题目要求的 `list[float]` 返回值，额外的逐 epoch 指标保存在 `model.training_history`。

学习时先在一个小批次上训练到低损失，证明模型和梯度管线可用，再训练完整 MNIST。
看训练/测试曲线的差距，并报告真实测试准确率；达到 70% 是实现正确性的最低检查。
记录的训练 epoch loss 是训练过程中各批次损失的样本加权平均；测试损失在 epoch 末以 eval 模式计算。
两者的测量时点不同，因此测试损失略低于训练损失并不一定是错误。

## ex4：先读计算路径，再做公平比较

每张图切成不重叠 patches，patch 投影为 token，再加可学习位置向量。
普通注意力本身不包含二维位置关系，因此需要显式加入位置信息。
最后对 token 特征求均值并输出 10 类 logits；题目允许 mean pooling，不必添加 CLS token。

Pre-LN Transformer 的路径为：

```text
x ← x + Attention(LayerNorm(x))
x ← x + MLP(LayerNorm(x))
```

残差分支的输入/输出形状必须相同。每个编码器块要创建自己的 MLP，不能无意共享一个实例。
注意力这里用 `nn.MultiheadAttention(..., batch_first=True)`，输入轴顺序为 `(B,T,D)`。

FFN 与 GLU 的差别：

```text
FFN:   W_out(GELU(W_in x))
GEGLU: W_out(GELU(W_gate x) * (W_value x))
SwiGLU: W_out(SiLU(W_gate x) * (W_value x))
```

GEGLU 保留基线的 GELU，可以观察增加门控的影响；SwiGLU 则进一步替换门控非线性。
乘法门控让两条投影分支相互调节。做对照时只能改变要研究的因素。

忽略 bias，FFN 大约有 `2·D·F` 个权重，GLU 有 `3·D·G` 个权重。
令 `G≈2F/3` 可以使预算接近。整数取整和 bias 会带来少量差异，要报告实际参数量。

控制数据划分、训练轮数、学习率、批量、dropout 和主干结构；每组都重置初始化和打乱数据的随机种子。
相同种子保证各自可重复，但不同结构消耗随机数的顺序不同，所以不会让不同模型权重完全相同。
三个种子只提供初步的稳定性证据：报告均值和样本标准差，观察同种子的配对差值，
不能凭单次最好结果或误差带重叠就宣布统计显著。

掌握标准：能用自己的话解释门控公式、2/3 宽度规则，并从曲线和多次运行结果支持结论。
如果结果不支持 GLU 更好，就如实讨论模型规模、训练预算和 MNIST 数据集的限制。

## 视频准备

建议用两页图，录制不超过 60 秒的 mp4：前 10 秒展示 ex3 损失曲线和最终准确率，
接着 35 秒说明 ex4 的控制条件、三组曲线和最终/最佳准确率、参数量；最后 15 秒讨论
多随机种子的变化、可复现性以及结论的适用范围。
具体数字必须来自 `results/` 的实际实验；可以用 `ex4_summary.csv` 制作表格。
最终按 README 只提交四个 ipynb 和自己的 mp4，辅助脚本和本指南用于学习及复现实验。
