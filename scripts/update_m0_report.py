"""Add the verified full-run-01 acceptance record to its Markdown report.

Run with the py312 interpreter from a terminal with write access to the report.
The initial baseline report remains as history. Repeated calls are idempotent.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MARKER = "## Full-run-01 验收更新 — 2026-11-17"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run = ROOT / "docs/validation/m0/full-run-01"
    commands = json.loads((run / "commands.json").read_text())
    if len(commands) != 7 or any(c["exit_code"] for c in commands):
        raise RuntimeError("The full run did not complete all seven steps successfully")
    suite = ET.parse(run / "suite.xml").getroot()
    suites = [suite] if suite.tag == "testsuite" else list(suite.iter("testsuite"))
    counts = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if counts != dict(tests=122, failures=0, errors=0, skipped=0):
        raise RuntimeError(f"Unexpected suite counts: {counts}")
    lines = [MARKER, "", "本地 conda py312 完整补验通过；初次受限环境记录保留为历史。",
             "测试产物：代码仓库 `docs/validation/m0/full-run-01/`。", "",
             "- 全部 7 个步骤 exit_code=0；pytest：122 passed，无失败、无跳过。",
             "- Nautilus 端到端测试通过。"]
    for name in ("demo_covariance_kernel", "demo_psf_photometry"):
        record = json.loads((run / "notebooks" / name / "execution.json").read_text())
        if record["status"] != "passed" or record["executed_code_cells"] != record["total_code_cells"]:
            raise RuntimeError(f"Incomplete notebook: {name}")
        lines.append(f"- {name}：{record['executed_code_cells']}/{record['total_code_cells']} 单元，"
                     f"{record['seconds']:.2f} s。")
    old = json.loads((ROOT / "docs/validation/m0/manifest.json").read_text())
    new = json.loads((run / "manifest.json").read_text())
    for key in ("head", "source_sha256", "input_sha256", "notebooks_sha256"):
        if old[key] != new[key]:
            raise RuntimeError(f"Baseline changed: {key}")
    lines += ["- HEAD、生产源码、真实输入与原始 notebooks 哈希均与初始记录一致。", "",
              "| 场景 | Mean acceptance | min(post-burn steps / tau) | tau_ok |",
              "| --- | --- | --- | --- |"]
    for name in ("example1_f200w_run1_broad", "example1_f200w_run2_tight", "example2_f200w"):
        with np.load(run / "notebooks/demo_psf_photometry" / f"{name}.npz", allow_pickle=False) as archive:
            ratio = float(np.min((archive["chain"].shape[0] - int(archive["burnin"])) / archive["tau"]))
            lines.append(f"| {name} | {archive['acceptance_fraction'].mean():.3f} | "
                         f"{ratio:.1f} | {bool(archive['tau_ok'])} |")
    lines += ["", "三组达到当前 50 tau 经验门槛；broad 组余量较小。该诊断不替代独立重复运行与 posterior calibration。",
              "pytest 的 5 个 warnings 为既有 photutils 弃用/空检测提示。",
              "notebook 使用 Agg 后端，本次确认数值执行，未逐图验证图像展示质量。", "",
              "**M0 基线 gate 可通过，可进入 M1。PSF 来源已由构建者确认：PSFEx 从 mosaics 恒星构建，按 ePSF 处理。",
              "中心、归一化、PSFEx 配置及实际 renderer 的恒星验证仍待完成；本次结果不能证明 Sérsic 科学渲染已验证。**", ""]
    update = "\n".join(lines)
    if args.dry_run:
        print(update)
        return
    original = args.report.read_text()
    if MARKER in original:
        print("Report already contains this update")
        return
    title, _, history = original.partition("\n")
    args.report.write_text(title + "\n\n" + update +
                           "\n## 初次受限环境记录（历史，非当前状态）\n\n" + history.lstrip())
    print(f"Updated {args.report}")


if __name__ == "__main__":
    main()
