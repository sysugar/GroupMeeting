### 本周工作内容
- **DeepSeek R1微调实战**
- **改开题报告**

1. **DeepSeek R1微调**
   - 数据集准备，使用medical-o1-reasoning-SFT数据集（CoT数据集）
[![pE1ctcq.md.png](https://s21.ax1x.com/2025/02/25/pE1ctcq.md.png)](https://imgse.com/i/pE1ctcq)
   - 创建有监督微调对象，开启微调
[![pE1cwHU.md.png](https://s21.ax1x.com/2025/02/25/pE1cwHU.md.png)](https://imgse.com/i/pE1cwHU)
   - 带入问题测试，保存模型权重
1. **改开题**
深度强化学习优化：针对现有训练过程得到金价预测模型参数效果不佳而导致金价预测精度低的问题，使用深度强化学习算法优化预测模型的更新策略。使用xLSTM-SA预测模型，在预测过程设计深度强化学习算法中的状态、动作、奖励值、损失函数等元素，提出一种结合深度强化学习算法的模型更新策略，优化预测模型的训练过程，从而提高金价预测精度。
- 数据收集与预处理 ：收集金价数据，对数据进行清洗、标准化和特征工程等预处理操作，将数据划分为训练集、验证集和测试集。
- xLSTM 模型训练 ：使用训练集数据对 xLSTM 模型进行训练，通过优化算法调整模型的参数，使模型能够较好地拟合历史数据。在训练过程中，可以使用验证集数据进行模型选择和超参数调优，避免模型过拟合。
强化学习环境搭建 ：根据前面设计的环境，搭建强化学习的训练环境。将 xLSTM 模型的预测结果作为环境的一部分，与强化学习算法进行交互。
- 强化学习模型训练 ：使用强化学习算法对模型进行训练，初始阶段可以采用随机策略或基于 xLSTM 模型预测结果的简单策略进行探索，随着训练的进行，模型根据奖励信号不断调整策略，逐渐学习到更优的预测和交易策略。在训练过程中，可以使用经验回放、目标网络等技巧来提高训练的稳定性和效率。
- 模型评估与测试 ：在训练完成后，使用测试集数据对模型进行评估和测试，计算模型的预测准确率、交易收益率等指标，评估模型的性能。如果模型性能不满足要求，可以返回调整模型的结构、参数或训练策略，进行进一步的训练和优化。


- 在论文《One Fits All: Power General Time Series Analysis by Pretrained LM》中，作者提出了使用预训练的语言模型（LM）进行时间序列分析的方法，并通过微调（Fine-tuning）来适应特定的时间序列任务。以下是论文中关于微调进行预测的主要方法和步骤：
   1. **模型结构和微调策略**
冻结预训练块：作者使用了预训练的Transformer模型（如GPT2），并冻结了其自注意力（self-attention）和前馈（feedforward）层。这些层包含了预训练模型中学习到的主要知识。
微调的组件：仅对输入嵌入层（input embedding layer）、归一化层（normalization layer）和输出层（output layer）进行微调。这些层负责将时间序列数据投影到预训练模型的维度空间，并进行最终的预测。
   2. **输入嵌入和归一化**
输入嵌入层：由于时间序列数据的维度可能与预训练模型的维度不匹配，作者使用线性探测（linear probing）方法来重新设计输入嵌入层，将时间序列数据投影到所需的维度。
归一化：为了进一步促进知识迁移，作者引入了简单的数据归一化块（如反实例归一化，reverse instance normalization），以标准化输入时间序列。
   3. **微调过程**
数据准备：在微调过程中，作者使用特定任务的数据集对模型进行训练。这些数据集包括时间序列的历史数据和目标数据。
训练配置：作者使用了较小的学习率和适当的正则化技术（如dropout）来防止过拟合。训练过程中，模型的参数会根据任务数据进行更新，以适应特定的时间序列任务。
   4. **实验结果**
性能提升：通过微调，GPT2-backbone FPT在多个时间序列任务中表现出色，包括分类、异常检测、插补、短期预测、长期预测、少样本预测和零样本预测等任务。例如，在长期预测任务中，GPT2(6) FPT在ETTh1数据集上相对MSE降低了9.3%。
跨领域适应性：作者还展示了模型在跨领域任务中的适应性，例如使用BERT和BEiT等预训练模型进行时间序列预测，结果表明这些模型也能取得较好的性能。
   5. **理论分析**
自注意力与PCA的联系：作者通过理论分析，证明了自注意力模块在预训练模型中执行的功能与主成分分析（PCA）相似。这种联系有助于解释预训练模型如何能够跨越不同领域进行有效的时间序列分析。
   6. **未来工作**
参数高效微调：作者计划通过参数高效的微调方法（如LoRA）来进一步提高模型性能。这些方法通过引入低秩矩阵来减少微调时的参数量，同时保持模型的性能。
探索Transformer的通用性：作者还计划从n-gram语言模型的角度进一步探索Transformer的通用性，以更好地理解其在不同领域中的适应性。
通过上述方法，论文展示了如何利用预训练的语言模型进行时间序列分析，并通过微调来适应特定任务，取得了显著的性能提升。

[![pE1cS6x.md.png](https://s21.ax1x.com/2025/02/25/pE1cS6x.md.png)](https://imgse.com/i/pE1cS6x)
[![pE1cpX6.md.png](https://s21.ax1x.com/2025/02/25/pE1cpX6.md.png)](https://imgse.com/i/pE1cpX6)

问题：
1.使用DeepSeek微调是用黄金价格数据进行微调吗？输入是文本还是时序数据？
2.深度强化学习作用？量化策略还是优化？