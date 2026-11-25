# `jwst_psfmc` — Sérsic / PSF+Sérsic 开发与科学验证路线图

> **本地实施状态（2026-11-23）**：M0–M9 已实现，验证入口见[验证日志](validation/LOG.md)。[M0 补验](validation/m0/README.md)记录了 extras、两本 notebook 及用户确认的 PSFEx ePSF 来源。
>
> **科学适用范围**：M4 的核心预算按已测配置成立，超预算边界仍保留。M7 的合成验收和[真实恒星与空天区验证](validation/EMPIRICAL_PSF_SKY.md)分别记录。F444W 30 mas 的亮点源与紧致弱宿主分解仍对 PSF 和协方差敏感。M9 完成模拟验证，M10 已完成 F200W 真实目标联合拟合和 pre-SN 共享参考图的实测验证。
>
> **审查后改进**：联合别名与曝光检查、稠密内存预检、渲染精度状态、后验溯源和 M7 自动验收的复测见[审查整改](validation/LOG.md#m0m9-审查整改)。
>
> 以下保留 v2 原规划正文及当时的设计依据。

## 0. 审阅结论：保留主线，调整前置条件和验收逻辑

**原 roadmap 总体合理，值得沿用。** 最有价值的是：保留 legacy PSF 工作流、解耦 renderer/likelihood/sampler、先数值验证再采样，以及逐阶段 gate。主要问题不是缺少更多模型，而是若干会决定架构的科学约定出现得太晚；一些测试会把“内部自洽”误当“物理正确”。

### 0.1 本版的关键修订

| 优先级 | 原方案的缺口 / 风险 | 本版决策 | 对应阶段 |
| --- | --- | --- | --- |
| P0 | 所有 oversampled PSF 都使用 convolution → block sum | 首先区分 pre-pixel PSF 与 effective PSF；像素响应只能计算一次 | M0 / §2 |
| P0 | legacy 点源严格相等与新的线性卷积 compact limit 可能冲突 | legacy adapter 保证旧结果；新点源和 Sérsic 共用物理响应算子 | M2 / M4 |
| P0 | 有限 PSF 被归一化后，其 flux 被叫作无限总通量 | 分开记录 PSF 归一化参考、PSF 支持域损失、Sérsic 无限积分和 stamp flux | §2.4 / M4 |
| P0 | mask/covariance 正确性主要留到最终验证 | 在 generic likelihood 前建立 dense reference；新 API 明确选择 exact 或 approximation | M2 |
| P0 | `Parameter(bounds)` 无法完整定义 prior 和 sampling transform | 首版明确 physical prior、坐标变换、Jacobian、Nautilus prior transform | M1 |
| P0 | 高 n / 小 re 只提示“加大 oversampling” | renderer 必须有经过测量的有效域；越界拒绝或使用验证过的 reference backend | M3–M4 |
| P0 | 固定真值 coverage 被作为普遍 Bayesian 正确性要求 | 分开做 SBC、固定真值 performance、模型错配 robustness | M7 |
| P1 | 统一像素百分比误差不对应推断偏差 | 同时报告 flux/像素误差、噪声加权误差和参数偏移 | §7 |
| P1 | I/O 与 diagnostics 进入较晚 | 结果 schema 从首次 generic inference 开始；收敛失败也是结果状态 | M5 / M8 |
| P1 | 所有结构参数默认自由，缺少科学使用梯度 | 从已知形态、固定 n、自由 n，再到 PSF+host；分清模型假设和数据约束 | M5–M6 |
| P1 | PSF mismatch、背景和邻源只是泛化提醒 | 加入恒星假宿主检验、背景敏感性、独立 PSF injection、邻源案例 | M6–M7 |
| P2 | 后续 multi-epoch 只概括地讨论相关性 | 显式处理共用 reference 导致的跨 epoch covariance；避免重复使用数据 | M9 |

**执行原则**：合并门槛与 science-ready 门槛分开。可以合并受限的实验性实现；不能因此宣称整个参数域已验证。兼容性不能成为复制已知数值问题的理由。

### 0.2 首轮范围与交付物

首轮支持：单波段、单张规则像素网格、局部近似平移不变 PSF、固定 Gaussian noise model；PSF-only、Sérsic-only、PSF+Sérsic；constant background；free/fixed/简单 equality ties；emcee，随后完成通用 Nautilus 适配。

首轮默认科学场景是 **science image 上的 galaxy/AGN/transient-host decomposition**。difference image 的点源继续走已有工作流；普通正 Sérsic 不应自动解释成 difference image 中的静态宿主，见 §12。

交付物：可用 API + 测试、两个可执行示例、带版本的结果格式、validation manifest/CSV/图表，以及按参数和数据条件列出的适用范围。多波段、多 epoch、GPU、任意 constraints、一般非平稳 covariance 不是首版的前提。

## 1. M0 — 锁定真实基线，先解决会影响设计的未知量

### 1.1 当前代码映射（已核对）

| 文件 | 当前行为 | 扩展边界 |
| --- | --- | --- |
| `psf.py` | Fourier circular shift；block sum；下采样末端强制奇数尺寸；crop fractions | 保留 legacy；新增具备显式 origin/response 的新算子，不把旧下采样直接用于 Sérsic |
| `covariance.py` | kernel FFT、频谱 floor、有效像素/fit_weight、whitening | 保留旧行为；新 noise API 增加明确语义和 dense 参考 |
| `mcmc.py` | `[flux, dx, dy, bkg]` 四维；emcee / Nautilus；data-derived prior bounds | 旧入口保持；新推断独立于固定四维和旧自动 prior |
| `io.py` | emcee / Nautilus NPZ，`allow_pickle=False` | 新 schema 独立版本化；保留旧读取 |
| `plot.py` | PSF triptych、chain、corner | 新函数按自由参数名称和结果状态工作 |
| `pyproject.toml` | v0.3.0；Python ≥3.10；`dev` / `nautilus` / `notebooks` extras | 不随意增加硬依赖；参考 renderer 可为 validation extra |
| `examples/` | FITS 在 `examples/data`、PSF 在 `examples/PSF`；不随 wheel 打包 | fast tests 使用小型合成输入；真实 FITS 用 integration tests |
| `.github/workflows/ci.yml` | Python 3.10 / 3.11；安装 dev、nautilus | 后续增加支持的较新 Python；optional dependency 缺失路径单测 |

不要从较早包装规划推断当前存在 `jwst_psfmc/data/`、`get_demo_data_path()`，也不要把历史记录中的问题不经检查地当成现存缺陷：当前 `run_mcmc()` 已有显式签名，未知关键字不会再被 `**kwargs` 静默吞掉。

### 1.2 本次实际完成的定点检查

环境：Python 3.12.11、NumPy 2.5.3、SciPy 1.18.1、Astropy 8.0.1、emcee 3.1.6、photutils 3.0.0。未安装 pytest / Nautilus，**本次未运行完整测试套件、notebooks 或新 Sérsic 实现**。

| 检查 | 实际结果 | 正确解读 |
| --- | --- | --- |
| White-noise likelihood，9×9、异方差 err | FFT 与直接 chi-square 的 log-likelihood 差为 0（本次浮点计算） | 基础 FFT 归一化在该案例一致 |
| 5×5 periodic kernel：中心 1、四近邻各 0.1 | FFT 与 dense quadratic form 相对差约 `3.0e-16` | 可作为以后 small-matrix oracle 的起点 |
| 同一 toy case mask 中心 1 pixel | zero-residual quadratic `24.0378192`；marginal quadratic `23.9904658` | 差异确实存在；该 0.197% 不是一般 mask 偏差上界，也不是参数 bias |
| 实际 example2 PSF，4×，native=10 / 12 | 下采样变为 9 / 11；保留约 97.411% / 98.847% 的 prepared PSF sum | 旧 bookkeeping 未反映这一步损失；不能外推成所有数据同样的光度 bias |
| 同一 PSF，native=11 | block sum 的保留比例约 1 | 合适的 legacy regression 场景 |
| example1/2 F200W kernels，9×9 grid | 谱最小值约 0.2988 / 0.2891；floor 比例均为 0 | 不能因为 synthetic kernel 失效就声称真实数据频谱有 clipping 问题 |
| example2 PSF metadata | shape 535×535；常见 BUNIT/OVERSAMP/PIXSCALE 字段未提供所需说明 | 单靠文件名和数组不能确定是否已包含 pixel/drizzle response |
| 随机数入口 | `prepare_for_fitting` 未提供 seed；`run_mcmc(seed=...)` 仅限 Nautilus | 新接口的可复现性必须覆盖初始化和 sampler 两部分 |

上述 toy noise 检查使用 `default_rng(20261117)`，先生成 9×9 Gaussian data 与 uniform(0.5,2) err，再生成 25 元 residual。完整 benchmark 应在实施 M0 时纳入仓库，不能依赖本次临时检查脚本。

### 1.3 M0 必做清单

- [ ] 记录 HEAD、branch、dirty diff、依赖版本和真实 FITS/PSF/kernel 的 SHA256；保留用户未提交修改。
- [ ] 在隔离开发环境安装 `.[dev,nautilus,notebooks]`，运行现有 suite，记录 pass/fail/skip 数。
- [ ] 使用 bright/faint/non-detection/signed-flux 四种情形建立 deterministic goldens：image、同一 theta 的 logL、prior 边界、PSF fractions。
- [ ] golden 里固定 theta，不把不受控的 MCMC chain 当逐元素参考；随机结果比较 MC uncertainty 下的统计量。
- [ ] 执行原两个 notebooks；记录未通过的原因和已有失败，不用新 feature 掩盖基线问题。
- [ ] 查清真实 PSF 的生成方法：星像重建方式、是否 ePSF、oversampling、pixel response、drizzle kernel/pixfrac/scale、中心定义、归一化参考。
- [ ] 写三份短 design decision：PSF/units、noise/mask、parameter/prior；把尚不能确认的事实写成 blocker。
- [ ] 已知 legacy 问题建单独 issue/修复 PR；新 API 默认拒绝会触发静默尺寸变化的配置。

**M0 Gate**：基线可复现；真实 PSF 的 response semantics 有证据或明确为 unknown。unknown 时可以推进 analytic PSF 的开发，**不能宣称真实 JWST PSF 的科学渲染已验证**。

开始实施时，在 checkout 内使用 Python ≥3.10 的隔离环境；以下是待执行命令，不代表本次已经完成：

```bash
git status --short
git rev-parse HEAD
python -m venv .venv-sersic
source .venv-sersic/bin/activate
python -m pip install -e '.[dev,nautilus,notebooks]'
python -m pytest -q
python -m pip freeze
```

如果当前 `python` 版本过旧，先选择满足要求的解释器。baseline 记录包括非零退出状态和 skip，不能只保存“最后一次跑绿”的摘要。

## 2. 前置科学契约：坐标、单位、PSF 和通量

### 2.1 坐标必须有明确公式

- 数组 `data[y, x]`；`+x` 为列索引增加，`+y` 为行索引增加。屏幕上向上/向下取决于 `imshow(origin=...)`，不是物理约定。
- native pixel center 使用整数索引；stamp geometric center 为 `((nx-1)/2, (ny-1)/2)`，偶数 stamp 也有定义。
- 新 source 的 `(dx, dy)` 相对此中心；外来 PSF 的 reference center 单独给出，不能默认等于 peak 或几何中心。
- native pixel j 的子像素中心为 `(j + (a+0.5)/s - 0.5)`，`a=0,...,s-1`。记录这条公式，避免 0.5 pixel 偏移。
- PA 从 +x 朝 +y 旋转，单位 degree，周期 180°，内部转换一次；与 WCS 天空 PA 分开。
- `re` 是 intrinsic 半长轴 half-light radius，native pixel；派生 `re_circ = re*sqrt(q)`，不把 convolved half-light radius 当 intrinsic `re`。
- source offset、PSF origin、padding 和 crop 必须共用一个坐标系统；禁止逐次 `argmax` recenter。

### 2.2 数据单位

新 API 首版优先接受 **flux per output pixel** 的数值，如 Jy/pixel；不在 renderer 内隐式推断单位。

若 SCI / ERR 是 MJy/sr：

\[
D_i^{\rm Jy}=10^6\Omega_i D_i^{\rm MJy/sr},\qquad
\sigma_i^{\rm Jy}=10^6\Omega_i\sigma_i^{\rm MJy/sr}.
\]

协方差左右各乘对应转换因子。必须核对最终 mosaic 的 pixel area/WCS 与处理历史；不能把某个 input detector pixel area 直接套到不同 output scale。若面积在 stamp 上变化，恒定 surface brightness 的 sky 变成 `b * Omega_i`，而不是恒定 Jy/pixel。首版可要求局部近似等面积，并记录近似误差。[R6]

legacy 的幅度定义保持不变；迁移时显式转换 data、err、flux、background 以及各自 prior。光度单位转换与 error inflation 分开；不要重复修正相关噪声。

### 2.3 PSF response：必须分两条渲染路径

“oversampled”只描述网格间距，不说明每个值代表什么。[R1–R2]

| 输入语义 | 正确 forward model | native 输出步骤 | 首版处理 |
| --- | --- | --- | --- |
| Pre-pixel PSF density / subpixel kernel mass | intrinsic source 与 PSF 卷积，再施加 pixel response | 对 native pixel 积分；uniform response 可 block sum | 首先在 analytic PSF 上实现 |
| Effective PSF / PRF，已包含目标像素响应 | intrinsic source 与 effective response 卷积 | 在 native pixel centers 采样；不要再对 native pixel 积分 | 真实 PSF 证实属于此类后实现对应分支 |
| Drizzled empirical response | 由完整成像/重采样流程定义的点源响应 | 按 response 定义输出；需验证局部 shift invariance 和相位 | 不凭文件扩采样倍率猜测 |
| Unknown | 未定义 | 不自动选择 | 新 science API 报错，要求提供定义；legacy 不变 |

连续表达式可帮助检查单位。若 `P` 是 pre-pixel density、`W_i` 是像素响应：

\[
M_i=\int W_i(x)\,[S*P](x)\,d^2x.
\]

若 `E(x)` 已给出“位于偏移 x 的点源在一个 output pixel 中的响应”，则：

\[
M_i=\int S(u) E(x_i-u)\,d^2u.
\]

后式对子像素的**源积分**仍需要 quadrature，但其输出不是需要再 block-sum 的 subpixel image。增加 oversampling 不能补救重复计算 native pixel response。理想 ePSF 在间距 `1/s` 的网格上，其数组和通常按 `s²` 缩放；不能无条件把该数组和当作 native integrated flux。[R1]

拟议 `PSFSpec` 至少记录 `response_kind`、sample spacing、reference center、normalization reference、输入/输出 pixel scale、支持域、来源/hash；真实 JWST 再记录 filter、field position、epoch、drizzle 信息和生成时使用的源 SED（若相关）。若不同 component 需要不同有效 PSF，首版只允许明确的近似或显式给出各自 PSF，不默默使用同一假设。

### 2.4 分开四种 flux 概念

1. `sersic.flux`：解析 profile 在无限平面上的积分，**在所声明 PSF 光度标定尺度上**。
2. `psf_normalization_reference`：输入 PSF 相对于什么 aperture/总通量归一化；有限文件的 sum 不是天然的 infinity EE。
3. `psf_support_fraction`：计算采用的有限 kernel 相对归一化参考保留多少；与旧 `psf_prepare_fraction` 区分。
4. `component_flux_in_stamp` / `component_flux_in_aperture`：逐个 posterior draw 由模型积分得到，包含形态与位置不确定性。

禁止每次把 cropped galaxy 重新归一化为 `flux`。PSF 可在声明的参考支持域上归一化一次；随后若截短 kernel，应保留丢失比例或扩大支持域。若 raw PSF 本身已经遗漏未知 wings，metadata 必须说明，不能声称恢复了真实无限总通量。一个 EE 标量可以校准总尺度，不能重建遗漏 wings 对空间分布的贡献。

用于 Sérsic 的 PSF 不能直接采用为 9×9 point-source fit 裁过的 prepared PSF；应从 raw PSF 独立准备。光度估计同时输出 aperture flux，可显著降低把高 n 外翼外推误当直接测量的风险。

### 2.5 兼容性与物理一致性分别验收

旧 `psf_model` 使用 circular shift + 裁剪/fraction 规则；新的 padded linear operator 不一定在所有边界条件下逐像素相同。

- 私有 `LegacyPointSourceAdapter` 直接复用旧路径，逐 theta 复现旧 image/logL，作为回归桥梁。
- 新 `PointSource` 与 `SersicSource` 共用同一个 `PSFOperator`，使 compact limit 有同一物理定义。
- 在无 wrap-around、单位/归一化/中心/response 一致的共同有效域，新旧输出应在约定容限内一致。
- 在旧路径的已知问题区间，明确记录有意差异，禁止为满足 golden 把旧错误移植到新 renderer。
- 旧公开函数签名与默认行为保持；若修复旧行为，另开明确说明影响的 PR/release。

## 3. 最小架构：够用即可，不先建大型框架

| 模块 | 新职责 / 修改 | 首次出现 |
| --- | --- | --- |
| `parameters.py` | 有序自由参数、fixed/tied、prior、transform、schema | M1 |
| `models.py` | PointSource / SersicSource / ConstantBackground / CompositeModel；render 与 components | M2 |
| `rendering.py` | `PSFSpec`、`RenderContext`、origin、response、padding、通用新 downsampling | M2–M4 |
| `sersic.py` | intrinsic intensity、归一化、数值积分；不读 data/noise/sampler | M3 |
| `noise.py` | diagonal / dense / FFT prepared noise；quadratic、logdet、whitening | M2 |
| `fitting.py` | prepare/evaluate/optimize、generic sampling adapter、FitResult | M2 / M5 |
| `mcmc.py` | 保留 legacy 入口；可复用小工具但不扩大四参数假设 | 全程 |
| `io.py` / `plot.py` | 通用结果 round-trip 和诊断，旧 API 继续可用 | M5 / M8 |
| `joint.py` | 将来按 observation 组合，首版不创建空架构 | M9 |

以上文件名为建议。若 `noise.py` 只需要几十行，可先作为内部模块；分文件应服从职责和测试边界，不以类/文件数量衡量完成度。

最小 contracts（设计示意，不是可执行 API）：

```python
values = parameters.decode(sample_vector)       # physical values, including ties
image = model.render(values, render_context)    # float64; exact data shape
parts = model.render_components(values, render_context)
log_l = noise.log_likelihood(data - image)      # handles validity exactly once
log_p = parameters.log_prior_in_sampling_space(sample_vector)
```

`Observation`/prepared fit 持有 data、err/mask、noise、render context、model specification；renderer 不访问观测噪声。`RenderContext` 的数组和缓存只读，cache key 包含 shape、origin、sampling、PSF hash、response、support 和 dtype。PSF FFT 可以缓存，依赖变化 morphology 的 image 不能误缓存。

**不要在本阶段实现**：任意 model plugin registry、表达式语言、自动 WCS 重投影、GPU backend、可微分整个 pipeline。保留清晰接口即可。

## 4. M1 — 参数、prior 与采样坐标

### 4.1 最小参数定义

每个参数记录：`name`、physical initial value、physical unit、prior 或 fixed value、可选 transform、可选 equality tie、periodicity。采样向量顺序只来自统一的 `free_names`，而不是 model 内硬编码。

| 情形 | 规则 |
| --- | --- |
| Free | physical prior 明确；initial 在 support 内；transform 可逆 |
| Fixed | 不增加 ndim；验证数值/物理合法性；不把 fixed prior 常数乘进模型 evidence |
| Tied | 如 `host.dx -> agn.dx`；只有 root parameter 被采样、被赋 prior |
| Conflicting tie | 循环、缺失目标、不同单位、fixed 与 tie 同时存在、矛盾 bounds 在 prepare 时拒绝 |
| All fixed | 允许 render/evaluate；采样入口返回清晰信息或拒绝 ndim=0，不制造假 chain |
| Periodic PA | wrap 不等于跨边界 mixing；summary 用周期统计，不能直接对 -89°/89°取线性中位数 |

首版支持 normalized Uniform、LogUniform（有限正上下界）、Normal / TruncatedNormal（有需要时）；不必第一天实现所有分布，但数据结构必须能表达完整 prior。新 Uniform log-density 为 `-log(hi-lo)`，legacy `log_prior` 返回 0 的语义继续保留。

### 4.2 transform 不应偷偷改变 prior

若 `z=log(re)` 而想保留 physical `re` 的 uniform prior：

\[
\log p_z(z)=\log p_r(e^z)+z.
\]

若想用 log-uniform `re`，那是另一个科学 prior，需要显式声明；不能仅因“在 log 空间好采样”而改变。不要对可为负的 difference flux 强行取 log。若后续用 `(e1,e2)` 替代 `(q,PA)`，必须重新推导 joint prior/Jacobian，不把独立 uniform e1/e2 当 uniform q/PA。

emcee 使用 sampling-space log prior + Jacobian + logL。Nautilus 使用同一 physical prior 的 inverse-CDF / unit-cube transform，并接收 **logL only**；不能重复加入 prior。prior transform 的量纲、边界和变换往返必须测试。[R4]

### 4.3 低 S/N、零分量与边界

- `q=1` 时 PA 不可辨识：允许固定 PA；自由 q 接近 1 时用 shape components 辅助诊断，勿把 PA 的宽 posterior 标成 sampler bug。
- `host.flux=0` 时 host morphology 不可辨识；null model 应移除 host，不能用一堆未约束 host 参数证明 PSF-only 的 evidence 等价。
- 对 science image 的非负 flux、difference image 的 signed flux，采用明确不同配置。
- prior bounds 不从同一 noisy stamp 的 peak 自动决定后再当独立 prior；旧策略保留在 legacy，新 API 要求显式先验或记录经验性构造。外部信息可用于 prior，优化初值不等于 prior。

### 4.4 验收清单

- [ ] free-vector ↔ physical-values 往返；顺序、单位、名称唯一性；invalid shape / NaN / bounds。
- [ ] fixed/tied 维度、循环检测、alias 只计一次 prior、root support 与 tie 一致。
- [ ] log/unlog Jacobian 与数值积分验证；同一 physical prior 用 linear/log parameterization 得到一致 posterior。
- [ ] prior transform 抽样的 CDF/quantile 与声明分布一致；不能只检查 samples 落在 bounds 内。
- [ ] seed 通过 `SeedSequence` 分配给初始化、noise、sampler；记录 child streams，不在每个 worker 重复同一个 seed。
- [ ] PA boundary、q≈1、全部 fixed、零 flux 的错误/结果状态有测试。

**M1 Gate**：prior 的数学语义可以独立于 sampler 说明和验证；legacy prior tests 不变。

## 5. M2 — 通用模型桥梁与可验证的 noise likelihood

### 5.1 先写 dense oracle，再抽取旧 helper

令 `D_sigma=diag(sigma_i)`，`R` 为 standardized residual 的 correlation matrix：

\[
C=D_\sigma R D_\sigma,
\quad \log L=-\tfrac12[r^TC^{-1}r+\log|C|+N\log(2\pi)].
\]

**异方差 err 本身并不使该模型近似**：只要真实 C 就是上述结构且 R 在目标边界条件下平稳，标准化是正确的。近似来自真实相关结构是否符合该模型、FFT 是否采用 periodic boundary，以及 kernel/err 估计误差。

完整矩形、R 为 block-circulant、NumPy 默认 FFT convention：

\[
r^TC^{-1}r=\frac1N\sum_k\frac{|\mathrm{FFT}(r/\sigma)_k|^2}{P_k},
\qquad \log|C|=2\sum_i\log\sigma_i+\sum_k\log P_k.
\]

使用完整 fft2 时每个 mode 按公式计一次；若改 rfft2，必须正确处理 interior / DC / Nyquist multiplicity。delta correlation 应给 `P=1`。相关噪声生成可用 `ifft2(fft2(white)*sqrt(P)).real`，并独立检查模拟方差和 ACF。

### 5.2 三个最小 backend 与默认策略

| backend | 数学语义 | mask | 用途 |
| --- | --- | --- | --- |
| `diagonal` | 独立 Gaussian，合法像素直接求和 | 精确删去无效像素 | 最小端到端路径、参考 |
| `dense` | 用户明确的 C，或按明确边界构造的 C | 使用 `C_VV` | 小 stamp 的科学路径及 oracle |
| `fft_periodic` | 完整矩形上的 circulant R | 不允许静默 zero-mask | 快速路径，需验证 periodic approximation |

legacy 保留原 `fit_weight` 处理。新 API 中有 correlated noise 且 mask 非空时：显式选择 dense，或使用已量化并带 `approximate` 标记的 legacy FFT 模式；不偷偷切换统计模型。dense 的内存约 `8*N_valid²` bytes，factorization 为 O(N³)；预先检查规模，不能在大 stamp 上无提示分配巨阵。

边界必须写进 backend 配置：dense 可以使用真实有限窗口的非周期 covariance；对照 FFT 时要另构造同一 periodic covariance。**用非周期 dense 和 periodic FFT 比较，本来就不应要求一致。** 可从更大平稳场裁出 noise 检验周期边界偏差。

### 5.3 mask、soft weights 与 log determinant

若只观测 V 中的像素，正确 marginal 使用：

\[
\log L_V=-\tfrac12[r_V^T C_{VV}^{-1}r_V+\log|C_{VV}|+N_V\log(2\pi)].
\]

把缺失 residual 设为 0 后用完整 inverse，使用的是 precision 的 V,V block；通常不等于 `inverse(C_VV)`。mask 个数少也不保证无影响，位置比占比同样重要。不要用 `residual * mask` 处理 NaN，因为 `NaN*0` 仍为 NaN；采用 indexing 或 `where`。

`fit_weight` 不是第二张 inverse-variance map。对严格正、固定的 weights，令 `A=diag(sqrt(w))`，旧 quadratic 可解释为改变后的 `C_eff=A^-1 C A^-1`；必须承认是在改变 noise model，且 normalized logL 要用相应 logdet。任意 aperture taper 不能直接声称为原观测的精确 likelihood；w=0 的 marginal 也不能用上述 zeroing 代替。首版新 science API 可先不开放 soft weights。

固定 C 的 posterior 可省略与参数无关常数，但应标记 `normalization='relative'`。若拟合 noise scale/Poisson variance，logdet 不可省略。不同 mask、数据、C 或 normalization 的 logL / evidence 不能直接相减。若 data/mask/C 完全相同，缺失的同一常数可在 Bayes factor 中抵消，仍须保证 proper priors 和一致实现。

### 5.4 covariance 与 error 的实际限制

- kernel 要 centrosymmetric、zero lag 与 normalization 明确、频谱正定。先测量再决定 regularization；记录 raw `P_min`、clipped fraction、floor 和修正量。
- 别用幅度极小的 positive floor 掩盖无效 covariance；支持域截断/窗口本身可能改变正定性。不同 floor 的敏感性要可报告。
- `R_ii=1` 或平均 P=1 是 ensemble normalization，**不意味着 whiten 后任意一张 image 的 sample variance 必然不变**。
- ERR 必须说明是 background-only、pipeline total，还是已做经验 scaling；source Poisson 不要重复计入。
- 从 blank sky 得到的 correlation 不必等于明亮 source 的 Poisson correlation。首版以 background-dominated / fixed-C 为主；bright-source case 作为限制或独立验证。
- estimated kernel 的不确定性首版可用多 sky patches/bootstrap sensitivity 量化；固定 kernel 的 posterior 应标为 conditional，不宣称已边缘化 kernel uncertainty。
- 用于估计 kernel/sky 的区域应与拟合 source 足够分离且记录选择方式；若从拟合 residual 自适应估计再固定，可能吸收模型错配。验证 estimated-kernel 工作流时，应重复 noise estimation 步骤，而不只固定一次估计的 kernel 再模拟。

### 5.5 M2 任务与 gate

- [ ] 先写 diagonal、small dense reference 和三个最小案例：white、periodic correlated、masked correlated。
- [ ] 通过 legacy adapter 抽取 residual-only helper，err/weight/valid 只应用一次；同 theta 严格回归。
- [ ] 实现 `CompositeModel` 分量求和、唯一名称、background 只加一次。
- [ ] new preparation 记录 invalid pixel 数量/位置；0 valid、错误 shape、非正 variance、非正定 covariance 清晰报错。
- [ ] covariance 不依赖 theta 时缓存 Cholesky / spectrum；不在每次 logL 重算。
- [ ] quadratics、logdet、whitening orientation 分别验证；Cholesky 使用 solve，不显式求逆。

**M2 Gate**：legacy regression 和数学 oracle 同时通过；新 API 的 noise/mask/weight 模式可从结果 metadata 唯一判断。此时才允许把 generic fitting 当作 Sérsic 的基础。

## 6. M3–M4 — Sérsic 渲染、卷积与可靠参数域

### 6.1 解析 profile

旋转坐标：`x'=dx*cos(pa)+dy*sin(pa)`，`y'=-dx*sin(pa)+dy*cos(pa)`；`R²=x'²+(y'/q)²`。

\[
I(R)=I_e\exp\{-b_n[(R/R_e)^{1/n}-1]\},\quad
b_n=\operatorname{gammaincinv}(2n,1/2),
\]

\[
F=2\pi q n I_eR_e^2 e^{b_n}\Gamma(2n)b_n^{-2n}.
\]

用中心幅度 `I0=Ie*exp(bn)` 的 log-space 形式更直接：

\[
\log I_0=\log F+2n\log b_n-\log(2\pi qn)-2\log R_e-\operatorname{gammaln}(2n).
\]

zero-flux component 单独返回零阵列，不计算 `log(0)`；首版物理 Sérsic 限定非负 F。`n>0`、`re>0`、`0<q<=1`；实际支持范围以数值验证为准。[R3]

椭圆 aperture 的解析 enclosed fraction：

\[
f(<R)=\operatorname{gammainc}(2n,b_n(R/R_e)^{1/n}).
\]

它是无 PSF 的**椭圆 aperture**关系，不是任意 square stamp 的精确 fraction；PSF 卷积后要重新积分。

### 6.2 先选 backend，避免同时自研所有难题

**建议起点**：NumPy/SciPy 作为 production renderer；高精度 adaptive quadrature 与解析 Gaussian 作为 reference；GalSim 作为 optional independent validation backend。先做小型 accuracy/runtime comparison 再决定是否更换 production backend。

首版可从 `n=0.5,1,2,4` 的代表点开发，再扩展连续 n；不能因为测试了 n=4 就宣称 `[0.3,8]` 全部可用。GalSim 的当前 Sersic reference 有其自身 n 范围限制（文档为 0.3–6.2）；对 n=8 不可将它当成已支持的 oracle。比较其 shear 后 radius 时，要把半长轴与面积保持的 half-light-radius 定义转换一致。[R5]

**最小安全分两层**：先发布固定 n / 已验证范围；free n 只有在整个先验支持域的渲染精度可控后开放。解析 profile 可以接受更宽参数范围，不等于 production renderer 已证明可处理。

### 6.3 高 n / compact source 的积分

- 不能只在 native pixel centers 采样；不能假设 4× 或 8× 固定足够。
- 对中心区域做稳定 quadrature，或使用已验证的 Fourier/analytic backend；外围可较粗采样，但过渡应连续。
- 每个 subcell 的源光度积分与卷积离散误差分开测量：准确积分源 subcell，并不自动保证 PSF 与 convolution 也充分采样。
- 避免在 MCMC 中根据 theta 突然切换 grid size / padding / lookup table，制造不光滑 likelihood。优先在 prepare 时根据 prior support 选定配置；若 adaptive，测量跨切换边界的连续性。
- finite prior 中若有不能可靠渲染的区域，应在 prepare 时拒绝或要求缩小有效域。不能在 sampling 中把 numerical failure 悄悄变成新的隐含 prior。
- 合法 theta 的意外渲染异常是实现错误，应暴露；真正 prior 外 theta 才返回 `-inf`。

### 6.4 convolution、padding 与计算规模

Pre-pixel 分支：intrinsic subpixel mass → padded linear convolution → native block sum → 定位 crop。effective-response 分支：source quadrature 与 response 卷积/积分 → native centers evaluation。两者遵循 §2.3，不能混用。

- 最初用 `fftconvolve` / direct convolution 做对照。偶数 kernel 的 `same` crop 可能引入半像素 convention 差异；显式 origin + full-convolution crop 更容易验证。
- 新 downsampling 精确保留 requested shape，包括偶数、矩形；禁止复用强制 odd 的 legacy routine。
- 按保留 PSF 支持域确定 source rendering domain：output stamp 加 kernel support / allowed shifts 即可。有限 kernel 下，不必为了接近 Sérsic 的无限总 flux 渲染所有远翼。
- 增加 PSF support 可能让 stamp 外 host light 散射进来，因此仍须扩大 support 做 convergence；无限 PSF wings 的截断另列误差预算。
- 误差分账：Sérsic integration、PSF sampling/interpolation、PSF support、FFT/crop、output quadrature；分别改变一项检查，不把所有变化混成“加大 oversampling”。
- FFT shape 可用 `next_fast_len`，但不能改变 origin/phase；full fft 的峰值内存要包含 complex arrays 和每个 worker 的副本。
- 所有缓存的键包括完整几何/PSF 配置；改变 PSF、oversampling、origin、support 后要失效。

### 6.5 M3–M4 deterministic tests

| Test | 独立参考 | 能发现的问题 |
| --- | --- | --- |
| `n=0.5` ellipse + Gaussian PSF | covariance matrices 相加；axis-aligned case 用 erf 精确 pixel integration | q/PA、卷积、像素响应、尺度 |
| `n=1`、`n=4` profile | 解析关系 / Astropy amplitude-converted profile | bn、归一化、单位；单独不足以验证 integration |
| 椭圆 aperture 的 curve of growth | incomplete gamma | F、re、q；区分截断误差 |
| delta / identity response | 无卷积 reference | origin、奇偶尺寸、block sum |
| asymmetric PSF + offset source | 小阵列 direct convolution | 卷积与相关误用、PSF 翻转、PA 符号 |
| compact Sérsic limit | 同一 PSFOperator 的新 PointSource | 重复 pixel response、中心/光度定义 |
| crop/stamp invariance | 固定大坐标域的重叠区域 | 局部重归一化、位置漂移、边界 |
| phase sweep | x/y 各多种 subpixel phase，含 0/0.25/0.5 | 周期光度漂移、pixel phase bias |
| integration / support refinement | s→2s、margin→2margin、reference tolerance 收紧 | 不收敛和假收敛 |
| square / rectangular、odd / even | 同坐标 reference | 下采样意外裁剪、半像素偏移 |

实际 JWST PSF 必须增加 empirical star recovery；由同一错误 renderer 注入再拟合并不能通过这一关。

**M3 Gate**：解析 profile、数值积分和 flux conventions 有独立参考。  
**M4 Gate**：两类 response 中已实现的分支各自通过测试；未实现的分支显式拒绝。发布一份 `validated_render_domain`，包含 n/re/q/phase、PSF 类别、sampling/support、noise-weighted error 与 runtime，不只是一个固定 oversampling 数字。

## 7. 验收误差预算：精度与科学用途关联

设 production 与 reference 的 image 差为 `delta_m`，在同一有效像素、同一 covariance 下定义：

\[
\epsilon_C^2=\delta m^T C^{-1}\delta m.
\]

这是确定性的**数值模型差异在噪声尺度上的大小**，不是某次数据的完整 `Delta chi²`；实际 logL 差还含 residual 与 delta_m 的交叉项。对高 S/N source，即使 0.1% profile error 也可能显著，因此必须记录 flux / noise level。

| 指标 | 建议起始目标（待开发时冻结） | 限制 |
| --- | --- | --- |
| Legacy adapter image / logL | image 约 `rtol=1e-11`；logL 用绝对尺度加相对尺度容限 | 固定同一计算定义；跨平台不要求 bitwise |
| Unconvolved analytic normalization | 约 `1e-3` 相对误差或更小 | 先扣除解析支持域截断，不能要求小 stamp sum=F |
| Noise-weighted renderer error | `epsilon_C <= 0.1`，即平方 <=0.01 | 针对声明的 science flux/noise 条件；加倍 S/N 需重评估 |
| Refinement posterior stability | 关键参数中心变化 <约 0.1 posterior SD，且区间变化与 MC error 相容 | 只适用于可辨识、充分采样案例；不是普遍数学保证 |
| Dense vs FFT，同一 periodic C | 接近 float64 舍入误差 | 排除不一致边界/weights 后再比较 |
| Runtime / memory | M0–M4 profiling 后设预算 | 不提前承诺任意图像 1 ms |

像素相对误差只在 reference 不接近零的区域报告；同时用 max absolute error / total-flux-scaled norm，避免 faint wings 除以零造成误导。

可选线性化诊断：`J=dm/dtheta`，`H=J.T C^-1 J`，当 H 可逆且局部线性合理，数值模型误差引起的 parameter displacement 近似 `-H^-1 J.T C^-1 delta_m`（delta_m 定义为 production−truth）。退化时不用该公式制造虚假的精确 bias 估计。

**门槛不能在看完失败后随意放宽。** 先预注册核心场景与误差预算；探索性第一批结果可调整设计，但调整原因、新阈值和重新验证都要记录。

## 8. M5 — 先完成可诊断的 single-Sérsic inference

### 8.1 推进顺序

1. **固定形态、只拟合 flux/background**：先与 linear Gaussian 或数值积分参考比较；验证参数、noise 和 sampler 连接。
2. **固定 n，自由 flux/re/centroid/q/PA/background**：先 noiseless optimum，再 white-noise posterior。
3. **自由 n**：只在 M4 有效域内开放；检查 n–re–flux–background 退化。
4. **Correlated noise**：同一个 source truth 换 known C，最后换 measured kernel；不要同时换模型和噪声。
5. **真实 science stamp**：M7 之后才能标注经过科学验证的使用域；先给 example，不先给泛化承诺。

所有 Sérsic 参数自由时为 **7+1=8** 维；fixed n 时为 7 维。low S/N 可固定外部约束的 morphology，但得到的是 conditional flux uncertainty；不能省略这个限定。

拟议公开入口保留原方案名称，降低迁移成本：`prepare_model_fitting()`、`run_model_mcmc()`；新增 `optimize_model_fit()` 和 `evaluate_model_fit()`。`run_model_mcmc` 虽沿用名称，必须说明可选择 Nautilus。以下只展示接口关系，不是当前可直接运行的示例：

```python
model = CompositeModel([
    SersicSource(name="host"),
    ConstantBackground(name="sky"),
])
fit = prepare_model_fitting(
    data=data_flux_per_pixel,
    model=model,
    parameters=parameters,
    psf=psf_spec,
    noise=noise_spec,              # carries err / covariance / valid mask
    render_options=render_options,
)
initial = optimize_model_fit(fit, starts=initial_candidates)
result = run_model_mcmc(
    fit, sampler="emcee", initial=initial, seed=42,
    sampling_options=sampling_options,
)
```

实现时不要同时接受两套互相矛盾的 `err/cov_kernel/mask` 与 `noise_spec`；如提供便捷 overload，必须验证只能选择一种来源。新配置拒绝未知关键字和不适用的 sampler options。

### 8.2 初始化不是推断

- 优化使用与 sampling 完全一致的 objective / priors；区分 ML 和 MAP。
- 采用少量科学上合理的多初值，尤其 n/re、host/PSF flux split；记录每次 objective 和失败状态。
- 借助固定形态的线性 flux/background 解初始化，比任意各分一半通量更可靠。
- walkers 在有效域内、非重合、数值 full rank；不要把大量 draws clip 到同一边界。
- 全局 optimum 不能证明 posterior 单峰；多启动和独立采样仍必要。
- 不为了寻找“更好结果”盲目扩 prior；先检查单位、response、background 和支持域。

### 8.3 emcee 与 Nautilus

emcee：

- [ ] seed 同时控制 walker initialization 和 sampler RNG；记录版本与 RNG state。旧 seed 语义继续单独维护。
- [ ] 检查选用 move 的 walker 数要求；默认 stretch move 通常至少 `2*ndim`，不能继续写死 48 或 64 对所有模型适用。
- [ ] 保留 `(step, walker, param)` 结构；burn-in 后再分析，不用 flatten 后的链估计 autocorrelation。
- [ ] 以 post-burn-in 长度相对于 tau、tau 随长度的稳定性、ESS/MCSE、独立 ensemble 的一致性诊断。`T > 50–100 tau` 可作起点，不是充分证明。[R7]
- [ ] acceptance fraction 只是提示；没有适用于所有 posterior 的固定“合格区间”。
- [ ] 同一 ensemble 的 walkers 不当作独立 chains 计算 R-hat；若需要跨链诊断，用独立初始化/RNG 的 ensembles 并明确处理方式。[R7]

Nautilus：

- [ ] 任意 ndim 使用 M1 同一 prior transform，periodic PA 的处理与库能力匹配。
- [ ] 保存 normalized log weights、log likelihood、weighted ESS、likelihood calls；summary/plots 必须使用权重。
- [ ] 比较若干低维可积模型及同一 science mock 的 emcee posterior；固定 data/prior/noise/renderer，差异与 MCSE 比较。
- [ ] 记录 evidence normalization 状态。首版可只输出 posterior，暂不提供 host-detection Bayes factor。
- [ ] optional dependency 不存在时，import package 不应失败；仅请求该 sampler 时给出安装说明。

multiprocessing：先有 serial 正确性；采用可序列化的顶层 callable / dataclass，测试 spawn（对 macOS/Jupyter 尤其相关）。`ncores=1` 是最小稳健路径；多 worker 的缓存复制、BLAS threads 和 pickling cost 计入 benchmark。

**M5 Gate**：固定形态的解析/数值参考 posterior 通过；single-Sérsic white/correlated 小型案例通过；FitResult 能记录 `converged / insufficient_sampling / numerical_failure`。固定 4000 steps 不构成 gate。

## 9. M6 — PSF+Sérsic：识别退化与假宿主

### 9.1 模型与维度

\[
M=F_{\rm point}\,\mathcal R[\delta_{x_p,y_p}]
 +\mathcal R[S(F_h,R_e,n,q,PA,x_h,y_h)]+B,
\]

其中 `R` 表示同一、定义明确的成像响应。独立中心为 **3+7+1=11** 个自由参数；tie 两个中心后为 9；再固定 n 则为 8。

AGN 可以把 coincident centers 作为一个待检验的 science model。transient-host 通常不应默认 tie：NEXUS 中的 off-nuclear transient 尤其如此。中心是否共享由科学场景决定，不由优化方便决定。

### 9.2 首版必做任务

- [ ] 拼接已验证 components，不写另一份专用 likelihood。
- [ ] 明确允许 point flux 为负的场景；host 默认为非负且物理形态有效。
- [ ] 实现 equality ties、fixed morphology 和 component removal；free/fixed/tied 的 metadata、summary、保存一致。
- [ ] 记录 flux–re–n–sky 与 point/host flux correlations；记录 prior-boundary pile-up 和 unresolved 标志。
- [ ] 根据 posterior samples 计算 host fraction、aperture flux、total flux 与 `re_circ`；不要把各参数 median 塞进非线性公式充当 posterior median。
- [ ] 比较 stamp 扩大、背景 prior 改变、固定/自由 n；高 n 外翼与背景退化时，明确哪些 flux 是外推。
- [ ] 每个 decomposition 至少输出 data、total、point、host、background、residual，以及 standardized/whitened residual。

### 9.3 必须包含的注入案例

| 案例 | 目的 | 预期输出 |
| --- | --- | --- |
| host flux 固定为 0，或移除 host | 检查 reduction 到新 PSF-only | 同一 signal/logL；不要求含未约束 nuisance 的 evidence 相同 |
| point flux 固定为 0 | reduction 到 Sérsic-only | 同一 signal/logL |
| tied / untied，truth 有/无 offset | 中心约束是否造成错误宿主结构 | residual 与偏差记录 |
| `Fpoint/Fhost = 0,0.5,2,10` | 覆盖主案例和高对比压力测试 | 明确失败域，不只展示成功图 |
| unresolved host / q≈1 | 可辨识性与 prior sensitivity | 上限/退化标记，而不是虚假的精确 re/PA |
| pure star，注入 PSF A、拟合 PSF B | PSF mismatch 导致假 host | false-host response 与敏感性 |
| host + background gradient / neighbor | 模型错配 | 残差结构、mask/扩大 cutout/联合邻源模型的处理建议 |

真实恒星上做 PSF+Sérsic 分解，是检查系统误差的重要 null experiment。使用注入 PSF 与拟合 PSF 相同的 mock 可以诊断 inference，不能估计真实 PSF mismatch 下的假宿主率。PSF 的 field position、epoch、stellar SED 与 AGN/host SED 差异应进入 sensitivity analysis；不必首版就实现复杂 PSF nuisance model。

**host detection 不等于非负 posterior 的 median>0。** 首版报告 posterior、upper limits 和 null-injection calibration。若引入 Bayes factors，应先验证 normalized likelihood、proper prior、boundary/null morphology 的模型定义；不能直接把相关像素数塞入普通 BIC 后称为严格显著性。

**M6 Gate**：reduction、维度和组分求和通过；至少一个已解析 host、一个强退化 case、一个 pure-star mismatch case 有可解释的结果。此时功能可标 experimental，不能据此概括全部 AGN contrast 范围。

## 10. M7 — 三条相互独立的科学验证线

### 10.1 A：数值正确性 / independent injection

同 renderer 注入再拟合先用于连接测试；随后换独立高精度 renderer，检查实际模型误差。两者分别标为 `self_consistency` 和 `independent_render`。noiseless recovery 用 deterministic objective 和有限、明确的 reference weights；不要把 err 设为 0 后要求 sampler 工作。

测试真值之外，还检查 posterior 典型区域与 prior 边界；仅 truth 点准确不足以保证整个 likelihood surface 正确。

### 10.2 B：计算校准与固定真值性能分开

**SBC（simulation-based calibration）**：从声明的 proper prior 抽 truth，按相同 generative model 模拟数据，再拟合，检查 rank / coverage 是否与预期相容。这检验 inference 与模型的一致性。[R8]

- [ ] 先低维 linear case，再固定 n Sérsic，再主 decomposition case。
- [ ] 选取接近独立的 posterior draws，或使用适合相关/weighted samples 的诊断；不能把所有相关 MCMC draws 当独立 ranks。
- [ ] 包含 flux、aperture flux、shape combinations、log-likelihood 等信息量不同的量；单一参数 rank 不足以排除所有实现错误。
- [ ] 使用 simultaneity-aware 图带/预注册统计量，避免大量逐参数 p-value 筛出偶然“失败”。
- [ ] SBC 通过不等于真实 PSF/noise/profile 正确；它与 A/C 不能替代。

**固定真值 injection-recovery**：测量指定 science 条件下 bias、interval width、68/95% coverage、upper-limit behavior、失败率和 prior sensitivity。Bayesian credible intervals 不保证每个固定真值都有名义 frequentist coverage；低 S/N、边界和强 prior 的偏离不自动证明 sampler 有 bug。

对可辨识、高 S/N、弱 prior 的主域可要求 bias 和 interval behavior 达到预注册科学预算；在 F=0、re 接近分辨率极限等情形，报告实际 operating characteristics。不能只留下成功收敛的模拟再计算 coverage：同时报告全部尝试数量、收敛比例、成功子集 coverage 及失败原因。

### 10.3 C：模型错配 / 真实数据 robustness

- [ ] 独立 PSF 或同一 PSF 的宽度、wings、ellipticity、centroid、颜色/field 变化；幅度来自真实 PSF 差异或标为 stress test。
- [ ] 不同 sky patches 估计的 kernels；err scaling；PSF support；stamp size；background constant vs small gradient。
- [ ] 噪声由更大非周期场裁出，对照 periodic approximation；mask 分别位于 source core、wings、边缘与成簇坏像素。
- [ ] nearby source 与非 Sérsic host 的残差；不要指望相关噪声 likelihood 吸收 astrophysical mismatch。
- [ ] real blank-sky injection / real star null，保留真实 noise 的非理想结构；加入 source shot noise 时不得从已含该项的 ERR 再重复生成。
- [ ] posterior predictive checks：whitened residual、二维 ACF、radial/azimuthal residual。拟合消耗自由度，残差不会自动等价于原始 independent noise；用模拟包络判断。

单个 scalar error inflation 一般无法同时复现 flux、re、n、q 对相关噪声的响应；要比较不同 likelihood，按各参数报告 uncertainty ratio，不能由点源 flux 的一个 effective factor 宣称形态误差也已校正。

### 10.4 可执行的 mock 设计

先用 **约 20–30 个分层代表点**，每点 5–10 次用于 debug；不要做全部维度笛卡尔积。建议在 YAML/JSON manifest 中给每点唯一 ID，并预先分 `core` / `boundary` / `mismatch`。

| 维度 | 起始设计值 / 覆盖要求 |
| --- | --- |
| Host known-shape S/N | 5、20、50；另加高 S/N stress case 检查 renderer error |
| `re` | 0.3、1、3 native px；同时记录 `re/FWHM`，不同滤镜才可比较 |
| n | 0.5 analytic oracle；science 1、2、4；更高 n 只在 renderer 有效域内 |
| q / PA | 0.4、0.7、1；0° / 30° / 接近 ±90°；圆形时不考核 PA recovery |
| Phase / center | 中心、quarter/half pixel、允许位置范围边缘；tie / offset |
| Point/host contrast | 0、0.5、2；10 作为难例；pure point 单独定义 |
| Noise | diagonal；known correlated；estimated kernel；large-field crop |
| Mask | none；core；wing；edge；cluster，不能只报告 mask 百分比 |
| PSF | analytic Gaussian、asymmetric toy、代表性真实 JWST、独立 mismatch |
| Sky / support | 固定与自由 sky；至少两种 stamp / margin；邻源独立 case |

这里的 known-shape S/N 可定义为 `F*sqrt(t.T @ C^-1 @ t)`，t 为单位通量模板；这是未边缘化形态/背景的 reference S/N，不是最终分解显著性。若 sky/其他线性项自由，另报告相应 marginalized flux S/N。

对 3–5 个核心 science cases 扩展到至少 100 realizations。若想把 nominal 68% coverage 的近似 95% CI 半宽压到约 5 个百分点，通常需要约 335 个独立 trials，而非 100；实际用 Wilson/exact binomial interval。100 次的 68% coverage 标准误约 4.7 个百分点，95% interval 半宽约 9 个百分点。

100 次适合早期校准，不适合验证极小 false-positive rates 或“3 sigma”尾部。upper limit 应注明 posterior quantile、prior 和 one-sided/two-sided convention：例如 Gaussian one-sided 3σ 对应约 99.865%，不是泛称 99.7%。不把 credible upper limit 自动解释成检测显著性。

### 10.5 结果文件与 acceptance

`validation/design.yaml` 固定设计；`validation/results.csv` 每行是一次尝试、一个参数/派生量，至少包括：

```text
case_id, realization_id, test_family, baseline_commit, code_commit,
truth, parameter_name, prior_id, psf_id, kernel_id, renderer_id,
noise_backend, boundary_mode, mask_id, seed_noise, seed_sampler,
median, q16, q84, q025, q975, ess, mcse, status, failure_reason,
render_error, runtime_s, peak_memory_mb
```

完整 priors/PSF/units/config 放 JSON manifest，CSV 用 ID 引用；truth/interval 单位一致。SBC ranks 可另表，不能硬塞进同一 coverage 字段。

最低图表：bias & standardized bias、interval width、coverage with CI、convergence/failure fraction、PSF mismatch 假宿主、渲染精度/耗时、代表性 posterior/residual。报告 posterior 是否 prior-dominated；不是所有 prior sensitivity 都能靠更长 chain 解决。

**M7 Gate**：A/B/C 各有证据，包含失败域；core cases 达到预注册标准，boundary/mismatch cases 的限制可见。只给“coverage 看起来不错”或一张漂亮 triptych 不足以 science-ready。

## 11. M8 — 结果格式、诊断、性能和发布

### 11.1 FitResult 从 M5 就开始存在

最小字段：

| 类别 | 必须保存的内容 |
| --- | --- |
| Parameters | `free_names`、physical samples、sampling-space 定义、units、priors、fixed/tied、periods |
| Sampling | sampler/version、chain shape 或 weighted samples、logL / logprior / logposterior 的区分、log weights、seed/RNG、burn-in、ESS/MCSE、停止原因 |
| Model | component names/types、geometry conventions、PSF response/origin/support/normalization、renderer settings |
| Data/noise | shape、unit、pixel area、mask、err/kernel/PSF/data hash、noise backend/boundary/normalization/floor |
| Provenance | package + schema version、commit/dirty state、依赖版本、可定位输入和配置 |
| Quality | numerical-domain、prior-boundary、unresolved、PSF-sensitive、approximate-mask、sampling-failed 等标志 |

NPZ 的 structured metadata 用 UTF-8 JSON string；不使用 pickle/object arrays，保持 `allow_pickle=False`。新增 `save_model_results()` / `load_model_results()`，不改旧 NPZ 语义。对未知 future schema 要报错或返回明确的有限读取模式，不默默猜测。

一个 hash 只能验证文件，不能复原文件；为可复现 run 提供 self-contained bundle（inputs 可选）或稳定输入定位与校验。若只存 JSON model spec，重建对象使用受控 type whitelist，禁止 eval 任意表达式。支持 model/component FITS 可选导出，切 stamp 后 WCS header 要正确调整 CRPIX，不能照搬全图 WCS。

### 11.2 绘图与科学摘要

- `plot_model_fit`：data / total / residual；明确 data units。
- `plot_decomposition`：point / host / sky；检查 sum=total。
- `plot_residual_diagnostics`：`residual/err` 与真正 whitening 分开命名；带 mask 时 dense Cholesky whitening 依赖像素排序，不直接把它 reshape 后的二维 ACF 当平移不变诊断。
- radial profile 必须注明 bins、ellipticity、PSF convolution、error 估计；correlated pixels 的 annulus error 不能简单 `std/sqrt(Npix)`。
- corner/trace 使用真实 free names 和 weighted samples；无 chain 的 Nautilus 不伪造 trace。
- 标注 plotted model 是 MAP、maximum likelihood、posterior median image 还是某个 draw；component-wise median image 的和不一定等于 total median image。要保证 decomposition 的加法一致，用同一 draw 或 posterior mean。
- summary 输出 aperture flux、外推 total flux、re semi-major/circularized；近圆 PA 使用 circular summary 或说明不可约束。

### 11.3 性能：在正确性后优化，但在大规模验证前测量

benchmark 固定 stamp 25/51/101 等代表尺寸、sampling、PSF support、n/re/contrast；记录 cold/warm render、logL、prepare、ESS/second、total wall time 和 peak memory。使用实际可行组合，不默认对每个 101×101 stamp 运行 dense。

优先顺序：缓存常量/PSF FFT → 合理 padding/FFT shape → 批量或避免重复分量渲染 → multiprocessing。验证优化前后 image/logL；不要先用 float32/粗 n 插值换速度再不记录误差。

**可选加速：线性 amplitudes**。固定 morphology 时，point/host flux 和 constant sky 对模型是线性的，可用 weighted linear solve 初始化与构造独立 posterior reference。后续可以 conditional sampling 或解析 marginalization，但：

- profile likelihood（每个 theta 只取最优 amplitudes）不等于边缘后验；不能直接替代完整 Bayesian sampling。
- proper Gaussian priors 下可推导闭式积分；有限 uniform/nonnegative priors 需截断积分，相关 determinant/normalization 不能丢。
- 首版不以此优化为 gate；先确保完整采样正确，再用同一 mock 对照。

### 11.4 CI 层次与发布

| 层次 | 内容 | 触发 |
| --- | --- | --- |
| fast | 解析 profiles、小矩阵 likelihood、parameters、origin、schema、legacy deterministic regression | 每 PR |
| integration | 已有真实 FITS、small end-to-end、旧新 load、optional sampler smoke、spawn smoke | 相关 PR / CI |
| science/slow | 独立 renderer、SBC、100+ injections、PSF mismatch、runtime report | 手动或 nightly；release 前冻结一次 |
| packaging | wheel/sdist build、从仓库外 import、optional dependency 缺失、examples 文件定位 | release PR |

新 synthetic unit tests 不应因为 `examples/` 不存在就整模块 skip。保留真实数据 integration tests，但不让它们取代解析测试。若标记 `slow`，`pytest -q` **不会自动跳过 slow**；必须在命令或配置中选择。

建议命令（实现 markers 后）：

```bash
python -m pytest -m 'not slow and not integration' -q
python -m pytest -m integration -q
python -m pytest -m slow -q
python -m pytest --cov=jwst_psfmc --cov-report=term-missing
```

CI 增加当前支持的 Python 版本，但测试矩阵选择与最低依赖兼容；保留 baseline environment 和一个较新 environment。完整模拟不在每个小 PR 重跑，只有相应 renderer/noise/prior/sampler 变更或 release 前重跑。

README 发布时写明：science 场景、单位、PSF 类型、validated domain、mask/noise assumptions、已知失效情形、计算成本。更新 CHANGELOG 与两处版本号，遵守仓库 `RELEASING.md`。功能增加不自动意味着 1.0；1.0 是 API 稳定承诺。

**M8 Gate**：旧 workflow/archives 可用，新结果能保存、恢复、复画；两个新 demo 可从头执行；科学声明均可追到 validation artifact。无新版本发布授权时，文档规划本身不执行 tag/publish。

## 12. 两类 science workflow 必须分开

### 12.1 原始 science image：推荐先做

对 AGN/host 使用 PSF+Sérsic；对 off-nuclear transient 使用独立中心的 point+host。结构复杂的 host、邻源、背景梯度先通过 residual 检查；首版允许受限 fit，但不要把 model mismatch 藏进更大的 error bar。

若 shape 从更深 reference 固定到单 epoch，标为 conditional inference。若 reference shape uncertainty 显著，可对其 posterior draws 重拟合/传播，或后续 joint fit；不能把相同 science data 既用于 prior 又在 likelihood 中重复使用。

### 12.2 Difference image：静态 host 不应再完整相加

若 `D_t = I_t - A_t I_ref`，其中 A 包含重采样、photometric scaling 和 PSF matching，forward model 应与同一线性处理相符：

\[
M_{D,t}=M_t-A_tM_{ref}.
\]

理想情况下静态 host 抵消。剩余 host residual 可能来自 PSF/calibration/astrometry mismatch；单个正 Sérsic residual 是经验近似，不能自动解释成 host 总光度。

独立 input science/reference 时：

\[
C_{D,t}=C_t+A_t C_{ref}A_t^T.
\]

如果有 shared exposures、reference 含当前 epoch 或其他相关处理，还需交叉项。difference flux 代表相对 reference 的变化；除非 reference 确定无 transient，否则不等于该 epoch 的绝对 transient flux。首轮保持 legacy PSF difference workflow，不顺手把复杂 subtraction operator 纳入开发范围。

## 13. M9 — 后续 multi-epoch joint fitting

同滤镜联合接口和显式块协方差已实现。模拟验证、适用域和用法见 [M9 joint fitting](validation/M9_JOINT_FITTING.md)。

后续 M10 在真实 NEXUS 多期背景上验证联合拟合的误差校准，并检查真实目标拟合需要的曝光来源与配准。见 [M10 实测预验证](validation/M10_REAL_EPOCHS.md)。

首选在独立 science images 上构建共享 host、epoch-specific transient/background：

\[
M_t=\mathcal R_t[S_{host}]+F_t\mathcal R_t[\delta_{SN}]+B_t.
\]

每 epoch 有自己的 PSF、noise、mask、单位、WCS/registration 和 calibration；同一滤镜可先共享 host 参数，多滤镜 morphology/SED 是独立后续课题。

- [ ] 引入 observation list 与 joint parameter aliases；不把共享参数复制采样。
- [ ] relative astrometry、flux calibration 的不确定性固定时要声明，必要时加 nuisance parameters。
- [ ] 对真正独立的数据 `logL=sum(logL_t)`；有效数据增加可以改善共同参数，但不能预设所有 nuisance/degenerate cases 都单调缩窄。
- [ ] 如果拟合多个共用 reference 的 difference images，独立 science epochs 下仍有：

\[
\operatorname{Cov}(D_t,D_u)=A_t C_{ref}A_u^T,\qquad t\ne u.
\]

- [ ] 采用 block covariance 或 latent reference 模型；不直接相加单 epoch likelihood。
- [ ] 若已有 raw epochs，就避免把这些 epochs 与由它们构造的 reference 当独立数据一起使用。
- [ ] pre-explosion flux=0 需物理依据；不按文件标签自动认定。

**M9 独立立项**，不阻塞首版。当前最值得预留的是 named shared parameters、per-observation context 和明确的数据 provenance，而不是提前实现 joint framework。

## 14. PR 顺序、产物和停止条件

沿用 M0–M9 名称方便追踪，但将旧 M7 的基础 noise 验证前移 M2，旧 M8 的 schema 前移 M5。原来的四个大 PR 拆成较小、可 review 的单元。

| PR | Milestone / 文件重点 | 必须交付的证据 | Gate 未达成时 |
| --- | --- | --- | --- |
| **PR 0** | M0：baseline、PSF audit、决策记录 | commit/env/hash、goldens、当前失败列表、PSF semantics | 继续 analytic mock；不接真实 PSF science claim |
| **PR 1A** | M1：`parameters.py` | free/fixed/tied、prior-transform/Jacobian tests | 不接通用 sampler |
| **PR 1B** | M2：legacy adapter、`noise.py`、minimal models | old logL regression、white/periodic/masked dense oracles | 修 noise/contract，不写 Sérsic inference |
| **PR 2A** | M3：`sersic.py`、reference integrator | analytic Gaussian/exponential、normalization、积分收敛 | 限制 n/re 域，不用更多 MCMC 掩盖误差 |
| **PR 2B** | M4：`rendering.py`、新 PointSource | response、origin、phase、padding、compact limit、valid domain | 缩小支持范围或更换 backend |
| **PR 3** | M5：generic inference、FitResult、最小 I/O | fixed-shape oracle、single Sérsic、seed/convergence、sampler parity | 保留 evaluate/render，不开放未验证 sampler |
| **PR 4** | M6：composition、ties、diagnostics | reduction、resolved+degenerate+null-star cases | experimental 标记，不宣布 host detection 可靠 |
| **PR 5** | M7–M8：science validation、性能、docs、release prep | A/B/C validation、两个 demo、round-trip、限制清单 | 不发布 science-ready 声明 |
| Future | M9 | independent/shared-reference joint tests | 独立排期 |

### 14.1 最先落实的十个小任务

1. 锁定 baseline 和 dirty diff，跑旧 tests，不改 science behavior。
2. 为 example PSF 填写 provenance/response checklist，不能确认的字段保留 unknown。
3. 添加 `test_legacy_adapter_matches_psf_model`，覆盖同 theta、不同 phase 和合法 stamp。
4. 添加独立 white / periodic / masked covariance oracle；mask benchmark 提前变成可重用测试。
5. 写 `ParameterSet` 的 physical priors、fixed/tied、transform 与 seed tests。
6. 实现 new PointSource 的 analytic-PSF response 路径和 origin tests。
7. 实现 n=0.5/n=1 intrinsic profiles，用解析 pixel integrals 校验。
8. 实现 Sérsic convolution，先做 phase/sampling/support convergence，再增加 n/re 范围。
9. 固定 morphology 拟合 flux/sky，与解析或数值 posterior 对照。
10. 解锁 fixed-n Sérsic，之后才是 free-n 和 PSF+host。

任务 3–5 可按代码依赖调整提交次序，但不跳过 gate。这是开发顺序，不是时间承诺。

### 14.2 失败时应该查哪里

| 症状 | 优先检查 |
| --- | --- |
| compact Sérsic 比 point source 更宽 | 重复 pixel response；PSF semantics；中心积分 |
| flux 随 stamp size 改变 | 局部重新归一化、PSF wings 标定、背景/高 n 外推；先区分数值变化与统计变化 |
| half-pixel phase 有周期 bias | origin、PSF sampling、ePSF 归一化、偶数 crop |
| white-noise logL 都不一致 | FFT 1/N、weights/err 重复应用、mask/单位 |
| masked correlated fit 过于自信 | zeroing 与 marginalization 混淆；kernel/floor；source core mask |
| re/n/flux 贴边 | 可辨识性、背景、PSF mismatch、prior support；不是先无限加长 chain |
| 同 renderer mock 全通过，真实星像假 host | independent PSF / response / SED / spatial variation |
| 修改 log(re) 后结果变了 | physical prior 是否改变、Jacobian 是否漏加 |
| sample 数很多但独立结果不同 | mode mixing、ESS/MCSE、未保存 weights、初始化与 RNG |
| larger padding 没有改善 | PSF 本身截短或类型错误；单纯扩大零区域不能补回信息 |

## 15. Definition of Done 与日志

每 PR 复制此清单，不要求未触及的完整 science suite 每次重跑：

```markdown
### PR validation
- [ ] 具体问题与新行为可用一个输入/输出例子说明
- [ ] 涉及的单位、坐标、prior、PSF/noise assumptions 已写清
- [ ] 相关 deterministic / regression tests 通过
- [ ] 旧 PSF/API/archive 行为的影响已检查
- [ ] numerical support domain 与错误处理明确
- [ ] 随机实验保存 seed、版本、truth、inputs hash 和失败状态
- [ ] 必要的 independent oracle / mock 已运行，未运行的检查明确列出
- [ ] 性能/内存有测量，优化没有改变 scientific result
- [ ] schema/docs/example 随接口更新
- [ ] Gate 达成；否则明确 experimental 限制和后续 issue
```

`docs/validation/LOG.md` 每次记录：日期、PR/M#、baseline/current commit 与 dirty state、环境、输入/PSF/kernel ID、configuration/prior ID、tests pass/fail/skip、最大 numerical error、bias/coverage with CI、ESS/MCSE、wall time/memory、已知失败、是否能进入下一阶段。

当前本地完成度：

- [x] **M0** 基线、真实 PSF 来源、extras 与原始 notebook 补验。
- [x] **M1** 参数、先验、坐标变换和共享参数回归。
- [x] **M2** legacy 回归与噪声、掩膜 oracle。
- [x] **M3** 内禀 Sérsic 的独立数值验证。
- [x] **M4** 两类响应的核心数值预算验证及边界风险记录。拟合可附独立参考来记录精度状态。
- [x] **M5** 固定 n 推断、随机种子、FitResult 与旧、新归档兼容。
- [x] **M6** 点源＋宿主分解以及假宿主、退化诊断。
- [x] **M7** 核心渲染、SBC、覆盖率、采样器自动验收与独立实测。压力用例单列。
- [x] **M8** 本地结果往返、绘图、示例、性能和打包检查。
- [x] **M9** 同滤镜联合接口、块协方差与模拟验证。
- [x] **M10（扩展）** 完成真实背景注入、F200W 真实目标联合拟合和系统误差诊断，以及两历元共享 pre-SN 模板的差分协方差验证。适用范围和仍缺的曝光清单见 [M10 报告](validation/M10_REAL_EPOCHS.md)。

**最终发布判据**：旧用户工作流有回归保护；新物理 renderer 有独立参照；posterior 算法与 science 模型分别验证；数据不支持的结构参数被如实报告；科学适用范围可由测试记录追溯。代码覆盖率、拟合残差漂亮或 sampler 返回成功，都不能单独替代这些要求。

## 16. 参考依据与证据边界

代码依据是本次本地 checkout，不是联网抓取成功的远端版本。源码定位：`psf.py` 的 `downsample_psf/prepare_psf_for_oversamp/psf_model`，`mcmc.py` 的 `log_likelihood_cov_prepared/prepare_for_fitting/run_mcmc/_run_nautilus`，`covariance.py` 的 `kernel_power_spectrum/prepare_covariance_terms`，以及现有 tests/CI。

外部资料于 2026-11-17 查阅；实现时锁定依赖版本，不能因为链接指向 stable 就假设本地旧版本 API 相同。下列资料用于数学/库行为核对；表中的工程阈值和 PR 切分是本路线图的设计建议。

- **[R1] Photutils — PSF / ePSF definitions**：说明 fine-grid ePSF 的每个值仍对应一个完整 detector pixel 的响应。  
  [参考链接](https://photutils.readthedocs.io/en/stable/user_guide/psf.html)  
  [参考链接](https://photutils.readthedocs.io/en/stable/api/photutils.psf.ImagePSF.html)
- **[R2] GalSim — pixel integration 与 `no_pixel`**：含 pixel response 的 empirical profile 不应重复 pixel convolution。  
  [参考链接](https://galsim-developers.github.io/GalSim/_build/html/gsobject.html)  
  [参考链接](https://galsim-developers.github.io/GalSim/_build/html/_modules/galsim/interpolatedimage.html)
- **[R3] Astropy Sersic2D**：profile、amplitude、ellipticity、angle 定义；不是默认的高精度 pixel integrator。  
  [参考链接](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Sersic2D.html)
- **[R4] Nautilus priors / API**：unit-cube transform 与 prior/likelihood 分工。  
  [参考链接](https://nautilus-sampler.readthedocs.io/en/stable/guides/priors.html)  
  [参考链接](https://nautilus-sampler.readthedocs.io/en/latest/api_full.html)
- **[R5] GalSim Sersic**：n 支持范围、truncation 与 flux/radius convention。  
  [参考链接](https://galsim-developers.github.io/GalSim/_build/html/gal.html)
- **[R6] JWST pipeline photom / resample**：SCI/ERR/variance 的光度转换与 resampled error 的构造。  
  [参考链接](https://jwst-pipeline.readthedocs.io/en/stable/jwst/photom/main.html)  
  [参考链接](https://jwst-pipeline.readthedocs.io/en/stable/jwst/resample/main.html)
- **[R7] emcee convergence**：autocorrelation、Monte Carlo error；同一 ensemble walkers 不独立。  
  [参考链接](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)
- **[R8] Talts et al. — Simulation-Based Calibration**：Bayesian computation 的模拟校准。  
  [参考链接](https://arxiv.org/abs/1804.06788)

后续可比较 GALFIT / IMFIT / GalSim 等工具，但只有在输入 PSF、像素响应、mask、sky、目标函数和 priors 可比时，其结果差异才有明确意义。外部 fitter 的 point estimate 不作为 Bayesian posterior coverage 的替代。

