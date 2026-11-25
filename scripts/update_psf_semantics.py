"""Update Markdown decisions after the producer's PSFEx/mosaic clarification.

Run with py312 and --docs pointing to AI_md_files/jwst_psfmc. --dry-run prints
the proposed update without writes. Archived numerical audits remain untouched.
"""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- psfex-mosaic-provenance:start -->"
END = "<!-- psfex-mosaic-provenance:end -->"
UPDATE = """## PSF 语义与 individual-star 验证约定 — 2026-11-17

构建者 Mingyang Zhuang 已确认：PSF 由 PSFEx 从对应 mosaics 的恒星构建，
按 **mosaic-level effective PSF (ePSF)** 处理。该信息的证据等级为
`user_report`，不是从简化 FITS header 中推导所得。一般 deep epoch 使用
40–50 颗亮星，wide epoch 使用 100–200 颗，基本覆盖所有亮星。这些是典型
epoch 的数量范围，不替换当前 example PSF header 中记录的 29 颗。
下文旧版本的 response semantics 为 unknown 的表述仅代表补充信息之前的状态，
当前以此更新为准；未核对的具体配置及验证结论不据此补写为已完成。

### 已知与待核对的边界

- 已知：来源为 PSFEx + mosaic stars；采纳 effective response 分支，响应语义不再为 unknown。
- 待核对：PSFEx 配置/版本、PSF_SAMP、实际导出尺度、reference center、归一化/aperture reference、空间变化模型及导出位置。
- effective response 包含输入 mosaic 的综合成像响应；不能据此分别反演 detector 或 drizzle kernel。
- 后续点源分支按 ePSF 定义插值求值；不重复进行 native-pixel block integration。
- Sérsic 与 ePSF 的卷积仍需对 intrinsic source 做正确 quadrature；这不等于再次对输出像素积分。
- legacy block-sum 保留为回归基线；对该 ePSF 的重复平滑风险需要实测，尚未量化其参数偏差。
- 既有 manifest/psf-audit 是运行时历史，不回写；新 audit 引用代码仓库中的 psf-provenance-user.json。

### 第一阶段：直接使用参与构建的 individual stars

可以直接使用，且应作为下一步的主要测试材料。将其标为
`training_star_renderer_check`：验证导出、坐标、采样和 renderer 一致性，
不称作 held-out validation。亮星数量多不使训练星变成独立样本。

对每颗未饱和、无明显邻源污染的星，在相同原始 mosaic 网格上，比较
legacy block-sum 与 ePSF 插值求值两条路径。两者使用同样的 flux/dx/dy/constant
background 自由度、mask、stamp 和有限孔径归一化参考；不要通过重心对齐后
叠加的星像替代原始 individual cutouts，否则会隐藏 subpixel-phase 问题。

若 PSFEx 包含空间变化，优先在每颗星的位置导出对应 PSF；若只有 central
PSF，先比较中心附近恒星，并单独记录到中心距离，避免把空间变化误认为
像素响应处理错误。记录 field position、subpixel phase、亮度/颜色和 epoch。
亮星的 residual 权重必须考虑适当的 source Poisson/模型误差，不能无条件
把仅由 blank sky 标定的 covariance 当作完整亮星噪声模型。

输出每颗星的径向/二维残差、core/wings mismatch、相对参考孔径的 flux
差异、位置差异和改变 stamp 后的稳定性；再按 pixel phase、field position
和亮度统计。两条路径都应使用足够支持域和明确的 origin。检查多颗恒星
而非单个例子；不预设 ePSF 路径必然获胜，不用这批训练星宣称泛化误差。

### 第二阶段：需要 science-ready 泛化结论时再交叉验证

不用强求另找足够亮且未用过的恒星。可以重建 PSFEx 做 spatial/phase
stratified 5-fold cross-validation：deep 每次约留出 8–10 颗，wide 留出
20–40 颗，具体以真实可用样本数为准，尽量兼顾空间、相位与亮度覆盖。
每轮只在训练折上构建/选择 PSF，并在该轮未参与构建的恒星上评价。
如果要估计整套流程的泛化表现，模型选择、归一化校准等也不得用验证折调优。
最终用于科学数据的 PSF 仍可用全部可用恒星重新构建。

在真实训练星上直接做 PSF+Sérsic null test 也可以作为内部诊断，但报告
应标明其训练样本身份；不能将由此得到的假宿主率称为独立估计。
"""


def replace_note(text):
    if START in text:
        before, _, rest = text.partition(START)
        _, found, after = rest.partition(END)
        if not found:
            raise ValueError("Unclosed PSF provenance section")
        return before + START + "\n" + UPDATE + "\n" + END + after
    title, _, rest = text.partition("\n")
    return title + "\n\n" + START + "\n" + UPDATE + "\n" + END + "\n\n" + rest.lstrip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = [args.docs / "m0/README.md", args.docs / "m0/ADR-001-psf-units.md",
             args.docs / "m0/LEGACY-ISSUES.md", args.docs / "jwst_psfmc_sersic_development_roadmap.md"]
    # Resolve every target before modifying any file; retain the existing text.
    for path in paths:
        path.read_text()
    if args.dry_run:
        print(UPDATE)
        for path in paths:
            print(f"Would update: {path}")
        return
    subprocess.run([sys.executable, str(ROOT / "scripts/update_m0_report.py"),
                    "--report", str(paths[0])], check=True)
    for path in paths:
        original = path.read_text()
        updated = replace_note(original)
        if path.name == "ADR-001-psf-units.md":
            updated = updated.replace(
                "Status: accepted for the new API design; real PSF response is an open blocker.",
                "Status: effective_psf adopted from producer confirmation; renderer validation and detailed metadata remain pending.")
        if updated != original:
            path.write_text(updated)
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
