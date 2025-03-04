## 1.14 组会
<font size = 4>

### 1. 本周工作内容
- **论文阅读xLSTMTime : long-term time series forecasting with xLSTM**
- **上周代码运行**
- **学习使用git和latex，latex格式修改**
### 2.论文内容
1. **问题**
    Transformer面临问题：高计算需求，难以捕捉时间动态和管理长期依赖性。
    LTSF-Linear凭借其简单的架构，明显优于Transformer
2. **xLSTM模型**
   - **sLSTM 稳定长短期记忆模型**
   1. 结合指数门控、记忆混合和稳定机制
   2. 模型架构
   [![pEiPN9J.jpg](https://s21.ax1x.com/2025/01/14/pEiPN9J.jpg)](https://imgse.com/i/pEiPN9J)
   - 遗忘门：控制上一个时间步长期记忆保留多少的程度（决定哪些信息应该从细胞状态中遗忘）
   - 输入门：控制该时间步新信息加入到长期记忆的程度（决定哪些新信息应该被写入细胞状态）
   - 输出门：获取长期记忆中隐藏的短期记忆（决定哪些信息应该从细胞状态中输出）
   - 隐藏状态：捕捉序列历史信息并将其传递到下一个步长，使模型能够捕捉时间依赖性
   - 长期记忆（细胞状态）：保留网络长期记忆
   - 归一化状态：归一化细胞状态
   - 指数门控：通过指数门控机制，sLSTM能够更精细地控制信息在细胞状态中的流动。具体来说，输入门和遗忘门的指数形式可以调节新信息的加入和旧信息的保留程度，从而更好地管理长期依赖关系。
   3. 稳定机制
   [![pEiP5Hf.jpg](https://s21.ax1x.com/2025/01/14/pEiP5Hf.jpg)](https://imgse.com/i/pEiP5Hf)
   [![pEiPq3j.jpg](https://s21.ax1x.com/2025/01/14/pEiPq3j.jpg)](https://imgse.com/i/pEiPq3j)
   结合了遗忘门和输入门的状态，通过特定的数学操作（如取对数和指数运算）来重新调整门控值，避免了在训练过程中可能出现的数值不稳定问题，如梯度爆炸或梯度消失。
   > 为什么加入这个可以稳定？
   > >原因：（1）指数函数具有良好的数学特性，如平滑性和单调性，这使得门控值的变化更加平滑，避免了剧烈的波动。例如，输入门和遗忘门的指数形式可以将门控值限制在合理的范围内，防止过大的更新或遗忘操作。（2）通过取对数和指数运算，sLSTM能够重新调整门控值的尺度，使其在数值上更加稳定。这种操作可以看作是一种动态的归一化过程，有助于保持细胞状态的数值稳定性，从而在整个训练和推理过程中维持模型的稳定性。
   4. slstm前向传播模块代码
   ```
   def slstm_forward_pointwise(
    Wx: torch.Tensor,  # dim [B, 4*H]
    Ry: torch.Tensor,  # dim [B, 4*H]
    b: torch.Tensor,  # dim [1, 4*H]
    states: torch.Tensor,  # dim [4, B, H]
    constants: dict[str, float],
   ) -> tuple[
    torch.Tensor,
    torch.Tensor,
   ]:
    _ = constants
    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states.view(4, states.shape[1], -1), dim=0)
    # raw = raw.view(-1, 4, -1)
    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)
    # with torch.no_grad():  # THE difference to maxg aka max_gradient (here max / max_static)
    logfplusm = m + logsigmoid(fraw)
    # 选择性更新稳定状态m
    if torch.all(n == 0.0):
        mnew = iraw
    else:
        mnew = torch.max(iraw, logfplusm)
    ogate = torch.sigmoid(oraw)
    igate = torch.exp(iraw - mnew)
    fgate = torch.exp(logfplusm - mnew)
    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / nnew

    # shapes ([B,H], [B,H], [B,H]), ([B,H],[B,H],[B,H],[B,H])
    return torch.stack((ynew, cnew, nnew, mnew), dim=0), torch.stack(
        (igate, fgate, zraw, ogate), dim=0
    )
   ```
   - **mLSTM 矩阵长短期记忆模型**
   1. 大量内存容量序列建模任务的选择。引入矩阵存储单元以及用于键值对存储的协方差更新机制，显著增加模型存储容量，门控机制与协方差更新规则协同工作，管理更新。
   2. 模型架构
   [![pEiP23d.jpg](https://s21.ax1x.com/2025/01/14/pEiP23d.jpg)](https://imgse.com/i/pEiP23d)
   3. 键值对存储的协方差更新机制
   [![pEiPoE8.jpg](https://s21.ax1x.com/2025/01/14/pEiPoE8.jpg)](https://imgse.com/i/pEiPoE8)
   - 查询（Query, q）：查询向量用于与键（Key, k）进行匹配，以确定哪些值（Value, v）与当前输入最相关。在mLSTM中，查询向量通过特定的线性变换从输入向量生成，用于后续的注意力机制计算。
   - 键向量用于与查询向量进行匹配，以确定哪些值向量应该被提取。键向量同样通过特定的线性变换从输入向量生成，用于计算与查询向量的相似度。
   - 值向量包含实际要提取的信息。在mLSTM中，值向量通过特定的线性变换从输入向量生成，根据查询和键的匹配结果，值向量被加权求和，以生成最终的输出。
   > qkv机制优势;
  通过注意力机制提高信息检索的效率。
  与矩阵记忆单元结合，通过协方差更新机制，增强模型的记忆容量。
  使得mLSTM中的操作可以并行化执行，加速训练和推理过程，减少计算复杂度。
   4. mlstm模块代码
   ```
   class mLSTMCell(nn.Module):
    config_class = mLSTMCellConfig

    def __init__(self, config: mLSTMCellConfig):
        super().__init__()
        self.config = config

        self.backend_fn = parallel_stabilized_simple
        self.backend_fn_step = recurrent_step_stabilized_simple

        self.igate = nn.Linear(3 * config.embedding_dim, config.num_heads)
        self.fgate = nn.Linear(3 * config.embedding_dim, config.num_heads)

        self.outnorm = MultiHeadLayerNorm(ndim=config.embedding_dim, weight=True, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        h_state = self.backend_fn(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=self.causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

   ```
1. **整体模型架构**
[![pEiPRgA.jpg](https://s21.ax1x.com/2025/01/14/pEiPRgA.jpg)](https://imgse.com/i/pEiPRgA)
1. **实验结果**
### 3.DAN代码运行结果
使用SFC_with_rain数据集，预测"2021-12-16 00:30:00"之后的流量
[![pEiEX7t.png](https://s21.ax1x.com/2025/01/14/pEiEX7t.png)](https://imgse.com/i/pEiEX7t)
[![pEiVCcQ.png](https://s21.ax1x.com/2025/01/14/pEiVCcQ.png)](https://imgse.com/i/pEiVCcQ)
[![pEiVVA0.png](https://s21.ax1x.com/2025/01/14/pEiVVA0.png)](https://imgse.com/i/pEiVVA0)
### 4.下周计划
   - [ ] Late格式修改
   - [ ] xLSTM代码运行
   - [ ] 看论文
</font>