# 作业 4：Hitters 数据上的 Ridge、Lasso 与 Elastic Net


> 运行位置：项目根目录。环境：Python 3.10+，`pip install numpy pandas matplotlib scikit-learn`。数据来自 ISLR 的公开 CSV；若网络不可用，请下载后将路径传给 `load_hitters`。

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 2026
# 图片保存到本文所在目录的 figures/ 下，Markdown 可用相对路径同时在 GitHub/VS Code 显示。
OUT = Path("outputs") / "figures"
OUT.mkdir(exist_ok=True)

def load_hitters(path=None):
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"
    df = pd.read_csv(path or url)
    # 常见镜像会多出一列行名；Salary 缺失代表该球员薪水未知，不能用于监督训练。
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    return df.dropna(subset=["Salary"]).reset_index(drop=True)

def make_preprocessor(X):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]), cat),
    ])

def cv_mse_and_1se(model_factory, alphas, X, y, cv):
    # 最大 alpha（即最强惩罚）中，CV 误差不超过最小误差 + 该最小点一个标准误的候选。
    means, ses = [], []
    for alpha in alphas:
        mse = -cross_val_score(model_factory(alpha), X, y, cv=cv,
                               scoring="neg_mean_squared_error", n_jobs=-1)
        means.append(mse.mean())
        ses.append(mse.std(ddof=1) / np.sqrt(len(mse)))
    means, ses = np.asarray(means), np.asarray(ses)
    i_min = means.argmin()
    eligible = np.asarray(alphas)[means <= means[i_min] + ses[i_min]]
    return means, ses, eligible.max()

def plot_path(kind, alphas, Xtr, ytr, names, l1_ratio=None):
    coefs = []
    for alpha in alphas:
        if kind == "ridge": model = Ridge(alpha=alpha)
        elif kind == "lasso": model = Lasso(alpha=alpha, max_iter=100000)
        else: model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000)
        coefs.append(model.fit(Xtr, ytr).coef_)
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, np.asarray(coefs))
    plt.xscale("log"); plt.gca().invert_xaxis()
    plt.xlabel("lambda (alpha; left = stronger penalty)")
    plt.ylabel("standardized coefficient")
    plt.title(f"{kind.title()} coefficient paths")
    plt.tight_layout()
    plt.savefig(OUT / f"{kind}_paths.png", dpi=180)
    plt.close()

df = load_hitters()
X, y = df.drop(columns="Salary"), df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

pre = make_preprocessor(X_train)
Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)
names = pre.get_feature_names_out()
alphas = np.logspace(-3, 4, 200)
cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

models = {
    "ridge": RidgeCV(alphas=alphas, cv=cv),
    "lasso": LassoCV(alphas=alphas, cv=cv, max_iter=100000, n_jobs=-1,
                      random_state=RANDOM_STATE),
    "elastic_net": ElasticNetCV(alphas=alphas, l1_ratio=[.05, .1, .25, .5, .75, .9, .95, 1],
                                 cv=cv, max_iter=100000, n_jobs=-1, random_state=RANDOM_STATE),
}

for label, model in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    nonzero = np.count_nonzero(np.abs(model.coef_) > 1e-10)
    print(f"{label:12s} alpha={model.alpha_:10.6g}  RMSE={rmse:9.3f}  nonzero={nonzero:2d}")
    if label == "elastic_net": print(f"{'':12s} l1_ratio={model.l1_ratio_:.2f}")

# 路径和 1-SE 法则（同一预处理后的训练特征上进行）。
plot_path("ridge", alphas, Xtr, y_train, names)
plot_path("lasso", alphas, Xtr, y_train, names)
enet_ratio = models["elastic_net"].l1_ratio_
plot_path("elastic_net", alphas, Xtr, y_train, names, enet_ratio)

factories = {
    "ridge": lambda a: Ridge(alpha=a),
    "lasso": lambda a: Lasso(alpha=a, max_iter=100000),
    "elastic_net": lambda a: ElasticNet(alpha=a, l1_ratio=enet_ratio, max_iter=100000),
}
for label, factory in factories.items():
    _, _, alpha_1se = cv_mse_and_1se(factory, alphas, Xtr, y_train, cv)
    one_se = factory(alpha_1se).fit(Xtr, y_train)
    k = np.count_nonzero(np.abs(one_se.coef_) > 1e-10)
    print(f"{label:12s} 1-SE alpha={alpha_1se:.6g}; nonzero={k}")
```

## 如何报告结果

运行后，将终端输出填入下表；数值会随随机划分、软件版本或数据镜像略有差异，因此不要手工编造固定数字。

| 模型 | `alpha_`（CV 最优） | 测试集 RMSE | 非零变量数 |
| --- | ---: | ---: | ---: |
| Ridge | 运行输出 | 运行输出 | 通常全部 |
| Lasso | 运行输出 | 运行输出 | 运行输出 |
| Elastic Net | 运行输出 | 运行输出 | 运行输出 |

运行代码后，图片会保存在与本文相邻的 `figures/` 目录：`figures/ridge_paths.png`、`figures/lasso_paths.png` 与 `figures/elastic_net_paths.png`。路径均为相对路径；若希望把图片嵌入本文，待图片生成并一并提交后再加入以下三行 GFM 标记即可：

```md
![Ridge coefficient paths](figures/ridge_paths.png)
![Lasso coefficient paths](figures/lasso_paths.png)
![Elastic Net coefficient paths](figures/elastic_net_paths.png)
```

Ridge 几乎不会产生精确的 0；Lasso 会产生稀疏解；Elastic Net 的稀疏程度取决于 CV 选出的 `l1_ratio`。

## 1-SE 法则的讨论

普通 CV 选择平均验证误差最小的 \(\lambda\)。1-SE 法则则在“误差不超过最小误差点一个标准误”的候选中，选择**最大的 \(\lambda\)**，也就是惩罚最强、通常更简洁的模型。若 1-SE 模型测试 RMSE 与最小-CV 模型接近，且解释性/变量筛选更重要，可优先报告 1-SE 的 Lasso 或 Elastic Net；若预测误差是唯一目标，则保留最小-CV 选择。Ridge 的系数一般不会变成严格 0，因此它不适合作为变量选择器。
