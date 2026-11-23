# 整网入口的资格边界

本表为源码审计，不是新性能结果。G1相关门通过后才运行整网测量。

| 路线 | 当前源码证据 | 本轮处置 |
| --- | --- | --- |
| MACE GitHub旧桥 | `../o2-hp-github-whole-20261121-r1/public_bridge.py`只替换首层四个scalar×angular路径，显式`parity="canonical"`；不是一般O3→O2完整卷积转换 | full-parity主比较为HOLD_MAPPING；不能直接改标签或只增加零通道后宣布全模型合格 |
| MACE c591b16 | `../hpx-custom-kernels-20261120-r1/mace_variants.py`沿用自然宇称checkpoint，冻结前缀/物种投影/首层receiver融合；旧伴随不含完整训练参数及二阶 | 仅作已冻结历史优化对照；新训练必须保持canonical原参数活跃并补全混合导数 |
| MACE旧整网runner | `../o2-hp-github-whole-20261121-r1/whole.py`设置Graph并调用`Replay`，大量旧case为静态图 | 新runner直接eager调用，不能复用旧Graph时间作新分母 |
| Eqx MACE | `upstream/eqx/equivariantx/models/mace/conversion.py`及`models/convolution.py`提供保持原模型接口的aligned O2卷积转换 | G1继承的Eqx只参与源/形状构造；正式对照必须另冻结本轮0ff7cad1395ae88d8aaf50633a04e2191b3f2c1c，验证映射、全部参数/二阶及实际cuBLAS路由 |
| V3 4b17046 | `../v3-so2-fp32-repair-20261121-r1/REPORT.zh-CN.md`已披露官方SO2/SO3表示，14个SO2Linear精度修复，未引入独立full-parity通道 | 核心运算次序可参考；full-parity后端映射及live权重二阶需重新证明，不能继承冻结输入VJP资格 |
| eSCN后端 | 原生SO2通道线性与O2 TP的权重约束/参数计数不同 | 同模型函数映射未证明前标HOLD，不比较各自原生架构的速度倍率 |
| TrimSO2 | 作者仓库本轮查询404，只有论文数据流依据 | HOLD_SOURCE，不能用自写复现署名作者实现 |

半键存储必须逐项保留原方向贡献的顺序/重复/周期自边和邻居截断结果。除N和实际方向贡献E外，独立记录物理半键B及未成互反对的方向边。构图与方向适配成本进入e2e边界；固定拓扑只作附录。所有权重/激活/归一化/损失/优化器与输入次序绑定原checkpoint，后端改变不能新增可训练自由度。

full parity作为后端资格需要完整的表示嵌入、提取、系数和权重映射证明，并核验所有现有可训练参数梯度。原checkpoint本身为自然宇称时保持其函数，不能宣称该checkpoint具有新增独立full-parity自由度。无法给出等价映射的模型/后端单元继续HOLD，其他合格单元独立推进。
