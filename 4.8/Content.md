## 本周工作内容
1. TimeMixer论文
2. TimesNet代码运行

## TimeMixer：Decomposable Multiscale Mixing for Time Series Forecasting

[![pEc7Eg1.png](https://s21.ax1x.com/2025/04/07/pEc7Eg1.png)](https://imgse.com/i/pEc7Eg1)

模型架构：

1. **Multiscale Time Series**：首先对输入序列进行不同程序的池化来下采样得到不同尺度的序列。
2. **Past Decomposable Mixing**：然后将每个尺度的序列都分解为趋势项（trend）和周期项（seasonal）（这里分解方法采用的是Autoformer中的方式，用一个大window size的滑动平均即可）。为了进行尺度间的交互，将每个尺度序列的seasonal项按照从下到上（bottom-up）的方式进行信息融合，将每个尺度序列的trend项按照从上到下（top-down）的方式进行信息融合。
3. **Future Multipredictor Mixing**：最后，将最后一个block输出的不同尺度的序列都用来预测，把所有尺度的预测结果加起来得到最终预测结果。

**Past-Decomposable-Mixing**

通过过去分解混合（PDM）块，将分解的季节和趋势成分分别混合成多个尺度。

具体来说，对于第 \( l \) 个PDM块，首先将多尺度时间序列 \( \mathcal{X}^l \) 分解为季节性部分 \( s^l = s_0^l, \ldots, s_M^l \) 和趋势部分 \( t^l = t_0^l, \ldots, t_M^l \)，通过来自Autoformer*（Wu et al., 2021）的序列分解块。考虑到季节趋势部分的独特属性，将混合操作应用于季节性和趋势性，以分别与来自多个尺度的信息进行交互。总的来说，第 \( l \) 个PDM块可以被形式化为：

\[ s_m^l, t_m^l = SeriesDecomp(\mathcal{X}_m^l), m \in \{0, \ldots, M\} \]

\[ \mathcal{X}^{l'} = \mathcal{X}^l - 1 + FeedForward(S-Mix((s_{m=0}^l)^M) + T-Mix((t_{m=0}^l)^M)) \]

其中 \( FeedForward(\cdot) \) 包含两个线性层，中间有GELU激活函数*，用于通道之间的信息互动。\( S-Mix(\cdot), T-Mix(\cdot) \) 分别代表季节性和趋势混合。


尺度间交互：作者用两种相反的流动方式来分别处理不同尺度的趋势项和不同尺度的周期项，如下图：


[![pEc7YKP.png](https://s21.ax1x.com/2025/04/07/pEc7YKP.png)](https://imgse.com/i/pEc7YKP)

- 对于周期项：如上图左侧，下面细粒度的seasonal序列用一个两层的MLP映射到和上面粗粒度的seasonal序列尺度对齐，然后相加即可得到融合后的结果，然后依此类推，把所有尺度的seasonal全部融合一遍。
  用bottom-up的流动方式原因：细粒度周期本身就包含了粗粒度周期，比如一个每小时采样的序列，周期严格是24，那么用每2小时、每半天、每一天的频率来采样该序列，则周期可以直接推算出来，分别是12，2，1。所以细粒度周期包含的信息多一些，用它来指导粗粒度周期会好一些。
- 对于趋势项：如上图右侧，其实是和周期项一样的处理方式，唯一的区别是方向是反的，是粗粒度逐渐映射到细粒度的。
  用top-down的流动方式原因：越是细粒度，趋势就包含越多的噪声和意想不到的变化，因此需要宏观趋势（粗粒度）来指导微观趋势（细粒度）。
注意，经过上述操作后，每个block得到的输出仍然是多个尺度序列的周期项和趋势项，只不过每个尺度的周期项和趋势项已经融合了其他尺度的信息。然后，把每个尺度的周期项和趋势项相加（趋势周期合并），即可得到每个尺度的未分解序列。再进行一个FFN变换，即可得到下一个block的输入。所以，每个block的输入是多个尺度的序列，依次进行趋势周期分解、尺度间交互、趋势周期合并，FFN，输出新的多个尺度的序列。

**Future-Multipredictor-Mixing**
那么对于最后一个block，它的输出也是多个尺度的序列，所以直接用多个predictor，对每个尺度的序列都映射到和预测范围的长度一致，然后所有尺度的预测结果相加即可得到最终的预测。每个predictor其实就是一个Linear，如下图所示：
[![pEc7tDf.png](https://s21.ax1x.com/2025/04/07/pEc7tDf.png)](https://imgse.com/i/pEc7tDf)

**结果对比：**
[![pEc7Nb8.png](https://s21.ax1x.com/2025/04/07/pEc7Nb8.png)](https://imgse.com/i/pEc7Nb8)

[![pEc7aVS.png](https://s21.ax1x.com/2025/04/07/pEc7aVS.png)](https://imgse.com/i/pEc7aVS)


### TimesNet运行结果（ETTh1）
[![pEgWODe.png](https://s21.ax1x.com/2025/04/09/pEgWODe.png)](https://imgse.com/i/pEgWODe)

[![pEgfFKS.png](https://s21.ax1x.com/2025/04/09/pEgfFKS.png)](https://imgse.com/i/pEgfFKS)

[![pEgfkDg.png](https://s21.ax1x.com/2025/04/09/pEgfkDg.png)](https://imgse.com/i/pEgfkDg)

[![pEgfVEj.png](https://s21.ax1x.com/2025/04/09/pEgfVEj.png)](https://imgse.com/i/pEgfVEj)

[![pEgfZUs.png](https://s21.ax1x.com/2025/04/09/pEgfZUs.png)](https://imgse.com/i/pEgfZUs)