"""Generate the continuation report from CPU audits; no new GPU claims."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    audit = json.loads((ROOT / "provenance/CLASSIFICATION-AUDIT.json").read_text())
    failures = json.loads((ROOT / "provenance/FAILURE-RECEIPTS.json").read_text())
    tavotto = json.loads((ROOT / "provenance/TAVOTTO-UPDATE.json").read_text())
    lines = ["# 6f641cf 后的 full-parity 修复续轮", "",
        "**当前尚未完成算术修复或H100复测。** GBU2连接不可用，未提交新GPU作业。"
        "已完成Tavotto升级、原始失败定位与汇总分类修正，准备了冻结输入的H100分段诊断。整网和训练暂停。", "",
        "## 已完成和更正", "",
        f"Tavotto插件 `{tavotto['plugin_version']}`、独立引擎 `{tavotto['installed_engine_version']}`。"
        "CPU115754离线升级成功，wheel按官方PyPI SHA256核验。当前MCP进程需重连才能加载新版。", "",
        "上轮将3个未执行的输入VJP从前向资格回执误判为数值失败，分类有误，现已修正。"
        "原始回执没有删除或改写。CPU115753独立复算得到：", "",
        f"`{audit['role_status_counts']}`。summary.csv、speedups.csv、numerics.csv与6f641cf逐字节一致，"
        "保持453个已验收曲线点和429个同方程配对，不把更正分类当作新增算子通过。", "",
        "## 真实数值失败定位", "",
        "8项真实失败都仅涉及第三输入radial_hidden的VJP，scale=1；前向、left和angular两路伴随通过。"
        "relative L2低于1e-5，未通过的是原逐分量门，不能用整体范数通过代替。", "",
        "| 实现 | 角阶 | 失败分量 | 原始失败检查 max_abs 最大值 | relative L2 最大值 |",
        "|---|---:|---|---:|---:|"]
    for item in failures:
        candidates = item["failed_receipts"]
        if not candidates:
            continue
        chosen = [c for c in candidates if "overnight-final" in c["receipt"]] or candidates
        values = chosen[-1]["failed_checks"]
        target = item["target"]
        assert {r["component"] for r in values} == {3}
        lines.append(f"| {target['role']} | {target['order']} | radial_hidden | "
                     f"{max(r['max_abs'] for r in values):.9g} | {max(r['relative_l2'] for r in values):.9g} |")
    lines += ["", "max_abs列仅概括失败窗口，不是单独验收阈值；逐元素容差随参考值变化。", "",
        "下一项diagnose_radial.py将保持整批E16384输入、权重和cotangent，分离：TP逐边权重伴随误差、"
        "固定权重伴随下改变矩阵执行行数的误差、以及参考线性层重放一致性。"
        "目前怀疑长K径向矩阵归约及批形状影响，但尚无H100分段回执，不能称为已证实根因。", "",
        "## 阶段门与资源", "",
        "| 阶段 | 当前状态 |", "|---|---|",
        "| Tavotto 0.21.3安装与版本核验 | 完成，运行中MCP待重连 |",
        "| 原始回执定位、3项误分类修正 | 完成，CPU审计通过 |",
        "| 8项算术修复、原FP32重验 | 未完成，H100诊断未运行 |",
        "| 15项构建超时处理 | 未开始，待前一门通过 |",
        "| 新同卡五轮计时及allocated/reserved | 未运行 |",
        "| 补齐角阶图及新版Tavotto最终导出 | 未完成，保留原453点和空缺 |",
        "| 整网、训练 | 暂停 |", "",
        "正式目标仍为GBU2 H100/torch2.11.0+cu128。直连18022、CPU节点直连、22端口及已固定密钥GBU转发均失败。"
        "不把CPU源码检查、安装成功或历史曲线当成新GPU验收。本轮GPU作业尚未递交，新增GPU消耗为0。"
        "准备的首个诊断限单卡30分钟，最多0.5GPUh，项目500GPUh上限不重置。", "",
        "719份上游算子源码与基线一致；移除只读诊断对象引用后，models.py的输入、权重、随机状态和参考表达式AST与6f641cf相同。"
        "这一审计仅证明源码准备，不证明任何新数值或性能门通过。", "",
        "[原始失败提取](provenance/FAILURE-RECEIPTS.json) · [分类审计](provenance/CLASSIFICATION-AUDIT.json) · "
        "[准备审计](provenance/PREPARED-QA.json) · [升级回执](provenance/TAVOTTO-UPDATE.json) · [续轮合同](CONTRACT-REPAIR.md)", "",
        "[冻结基线完整角阶报告](../o2-hp-angular-operators-20261121-r1/REPORT.zh-CN.md)。"
        "本轮尚无新性能数据，原基线报告及最终图保持原位。"]
    (ROOT / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")
    (ROOT / "STATUS.zh-CN.md").write_text("# 当前阶段\n\nH100连接阻断；8项真实数值失败尚未修复，15项构建超时待后续处理。\n\n"
        "Tavotto插件/引擎0.21.3安装完成，CPU分类修正及源码冻结审计通过；453个有效点的数值/性能/显存CSV未变。\n\n"
        "无新增GPU、整网或训练作业。详见[报告](REPORT.zh-CN.md)和[续轮合同](CONTRACT-REPAIR.md)。\n")
    print("Updated repair report: arithmetic/H100 acceptance still pending", flush=True)


if __name__ == "__main__":
    main()
