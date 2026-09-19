from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_carseats() -> pd.DataFrame:
    """优先从 ISLP 读取教材数据；若不可用，尝试 seaborn 的同名数据集。"""
    try:
        from ISLP import load_data

        return load_data("Carseats").copy()
    except ImportError:
        try:
            import seaborn as sns

            return sns.load_dataset("carseats").copy()
        except Exception as exc:
            raise RuntimeError(
                "未能加载 Carseats。请安装 ISLP：pip install ISLP，"
                "或提供含该数据集的 seaborn 环境。"
            ) from exc


def main() -> None:
    data = load_carseats()
    # 明确基准组：设为 Bad；回归输出中的 ShelveLoc[T.Good] 即 Good 相对于 Bad。
    data["ShelveLoc"] = pd.Categorical(
        data["ShelveLoc"], categories=["Bad", "Medium", "Good"]
    )
    model = smf.ols(
        "Sales ~ Price + Income + Advertising + C(ShelveLoc, Treatment(reference='Bad'))",
        data=data,
    ).fit()

    print("=" * 72)
    print("OLS 拟合报告")
    print(model.summary())
    print("\nShelveLoc 的基准组：Bad（货架位置差）。")

    good_name = "C(ShelveLoc, Treatment(reference='Bad'))[T.Good]"
    good_effect = model.params[good_name]
    print(
        f"ShelveLoc=Good 的系数为 {good_effect:.4f}：在 Price、Income、Advertising "
        f"相同条件下，Good 货架位的预测 Sales 比 Bad 货架位平均高 "
        f"{good_effect:.4f}（Sales 的数据单位，通常为千件）。"
    )

    # VIF 必须以与模型完全相同的设计矩阵计算；截距的 VIF 不用于诊断。
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    vif = pd.DataFrame(
        {
            "variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    vif = vif.loc[vif["variable"] != "Intercept"].reset_index(drop=True)
    print("\n各解释变量（含 ShelveLoc 虚拟变量）的 VIF：")
    print(vif.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    max_vif = vif["VIF"].max()
    if max_vif < 5:
        verdict = "最大 VIF < 5，通常认为没有明显多重共线性风险。"
    elif max_vif < 10:
        verdict = "最大 VIF 在 5 到 10 之间，存在中等共线性风险，建议进一步关注。"
    else:
        verdict = "最大 VIF ≥ 10，通常认为存在较强多重共线性风险。"
    print(f"\n诊断结论：{verdict}")


if __name__ == "__main__":
    main()
