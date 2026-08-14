# -*- coding: utf-8 -*-
"""
Task7 分析代码：NBA球员性价比分析（2024-25赛季）
====================================================
输入：nba_2024-25_cleaned.csv（Task6清洗产出，325人×47字段）
输出：6张图表png + 全部统计检验结果（控制台）
运行：python 姓名+Task7+分析代码.py
依赖：pandas numpy scipy statsmodels matplotlib
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

BLUE, RED, GRAY, GREEN = "#4472C4", "#C00000", "#8496B0", "#548235"

df = pd.read_csv("nba_2024-25_cleaned.csv")
df["lnSalary"] = np.log(df["Salary"])
print(f"[数据] {df.shape[0]}名球员 × {df.shape[1]}字段")

# ============ 1. 描述性统计 ============
cols = ["Salary_M", "PTS", "TS_pct", "EFF", "WinScore_48", "Age", "MP"]
desc = df[cols].describe(percentiles=[.25, .5, .75]).T
desc["skew"] = df[cols].skew()
print("\n[1] 描述性统计：\n", desc.round(2))
print(f"薪资右偏：均值{df['Salary_M'].mean():.2f}M / 中位数{df['Salary_M'].median():.2f}M "
      f"= {df['Salary_M'].mean()/df['Salary_M'].median():.2f}")

# ============ 2. 相关性分析 ============
print("\n[2] 与ln(薪资)的Pearson相关：")
for c in ["PTS_36", "TS_pct", "EFF", "WinScore_48", "Age", "MP"]:
    r, p = stats.pearsonr(df["lnSalary"], df[c])
    print(f"  {c:12s} r={r:+.3f}  p={p:.2e}")

# ============ 3. 回归建模（RQ1：残差识别高估/低估）============
# 基准模型：ln薪资 ~ 单位时间效率 + 得分产量 + 年龄
X = sm.add_constant(df[["WinScore_48", "PTS_36", "Age"]])
m1 = sm.OLS(df["lnSalary"], X).fit()
print(f"\n[3] 基准回归 R²={m1.rsquared:.3f}：")
for v in ["WinScore_48", "PTS_36", "Age"]:
    print(f"  {v:12s} coef={m1.params[v]:+.4f}  p={m1.pvalues[v]:.3f}")

# 控制合同类型（新秀红利）
X2 = sm.add_constant(df[["WinScore_48", "PTS_36", "Age", "rookie_suspect"]])
m2 = sm.OLS(df["lnSalary"], X2).fit()
print(f"控制新秀合同后 R²={m2.rsquared:.3f}，"
      f"rookie系数={m2.params['rookie_suspect']:.3f}(p={m2.pvalues['rookie_suspect']:.1e})"
      f" → 新秀薪资低约{(1-np.exp(m2.params['rookie_suspect']))*100:.0f}%")

# 稳健性口径：主要轮换 MP≥20
core = df[df["MP"] >= 20].copy()
Xc = sm.add_constant(core[["WinScore_48", "PTS_36", "Age", "rookie_suspect"]])
mc = sm.OLS(np.log(core["Salary"]), Xc).fit()
core["dev_pct"] = (np.exp(mc.resid) - 1) * 100
print(f"\n主要轮换口径(n={len(core)}) R²={mc.rsquared:.3f}")
print("  最被高估Top5：", ", ".join(core.nlargest(5, 'dev_pct')['Player']))
print("  最被低估Top5：", ", ".join(core.nsmallest(5, 'dev_pct')['Player']))

# ============ 4. RQ2 位置溢价 ============
print("\n[4] 位置分组：")
print(df.groupby("Pos").agg(人数=("Player", "count"),
      薪资中位数=("Salary_M", "median"),
      WinScore48中位数=("WinScore_48", "median")).round(2))
groups = [g["lnSalary"].values for _, g in df.groupby("Pos")]
H, p_kw = stats.kruskal(*groups)
print(f"Kruskal-Wallis检验：H={H:.2f}, p={p_kw:.3f}")
Xd = pd.get_dummies(df[["WinScore_48", "PTS_36", "Age", "rookie_suspect", "Pos"]],
                    columns=["Pos"], drop_first=True).astype(float)
md = sm.OLS(df["lnSalary"], sm.add_constant(Xd)).fit()
sig = [c for c in Xd.columns if c.startswith("Pos_") and md.pvalues[c] < 0.05]
print(f"控制表现与合同后，显著的位置虚拟变量：{sig if sig else '无（位置溢价不显著）'}")

# ============ 5. RQ3 顶薪错位 ============
df["dev_full"] = (np.exp(m2.resid) - 1) * 100  # 全样本模型残差→偏离百分比
max_p = df[df["Salary"] > 4e7].copy()
max_p["dev_pct"] = max_p["dev_full"]
q75 = df["WinScore_48"].quantile(.75)
t, pv = stats.ttest_ind(max_p["WinScore_48"], df[df["Salary"] <= 4e7]["WinScore_48"])
print(f"\n[5] 顶薪档(>4000万){len(max_p)}人：薪资均值{max_p['Salary_M'].mean():.1f}M，"
      f"WinScore/48中位数{max_p['WinScore_48'].median():.1f}")
print(f"  效率进入全样本前25%的仅{(max_p['WinScore_48']>=q75).sum()}/{len(max_p)}人；"
      f"薪资高于预测50%以上的{int((max_p['dev_pct']>50).sum())}/{len(max_p)}人")
print(f"  顶薪档vs其余球员 WinScore/48：t={t:.2f}, p={pv:.4f}")

# ============ 6. 图表输出 ============
# 图1 薪资分布
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.hist(df["Salary_M"], bins=30, color=BLUE, edgecolor="white")
ax.axvline(8.0, color=RED, ls="--", lw=1.2)
ax.text(8.6, ax.get_ylim()[1]*0.9, "中位数 8.0M", color=RED, fontsize=9)
ax.axvline(13.03, color=GREEN, ls="--", lw=1.2)
ax.text(13.6, ax.get_ylim()[1]*0.72, "均值 13.0M", color=GREEN, fontsize=9)
ax.set_xlabel("2024-25赛季薪资（百万美元）"); ax.set_ylabel("球员数")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig1_salary_dist.png", dpi=160); plt.close()

# 图2 相关性热力图
corr_cols = ["Salary_M", "PTS_36", "TS_pct", "EFF", "WinScore_48", "Age", "MP"]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
cm = df[corr_cols].corr()
labels = ["薪资", "每36分钟得分", "TS%", "效率值EFF", "WinScore/48", "年龄", "场均时间"]
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(7), labels, rotation=30, ha="right"); ax.set_yticks(range(7), labels)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cm.values[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(cm.values[i, j]) > 0.55 else "#333", fontsize=8.5)
plt.colorbar(im, shrink=0.8)
plt.tight_layout(); plt.savefig("fig2_corr_heatmap.png", dpi=160); plt.close()

# 图3 新秀红利
fig, ax = plt.subplots(figsize=(8.5, 4.2))
for flag, color, name in [(0, BLUE, "常规合同"), (1, RED, "疑似新秀合同")]:
    sub = df[df["rookie_suspect"] == flag]
    ax.scatter(sub["PTS_36"], sub["lnSalary"], s=14, alpha=0.55, color=color, label=name)
    xs = np.linspace(sub["PTS_36"].min(), sub["PTS_36"].max(), 50)
    b1, b0 = np.polyfit(sub["PTS_36"], sub["lnSalary"], 1)
    ax.plot(xs, b1*xs+b0, color=color, lw=1.8)
ax.set_xlabel("每36分钟得分"); ax.set_ylabel("ln(薪资)")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig3_rookie_gap.png", dpi=160); plt.close()

# 图4 主要轮换高估/低估Top8
top_over = core.nlargest(8, "dev_pct"); top_under = core.nsmallest(8, "dev_pct")
fig, ax = plt.subplots(figsize=(8.5, 4.6))
names = list(top_under["Player"])[::-1] + list(top_over["Player"])[::-1]
vals = list(top_under["dev_pct"])[::-1] + list(top_over["dev_pct"])[::-1]
ax.barh(names, vals, color=[GREEN]*8+[RED]*8, height=0.6)
ax.axvline(0, color="#333", lw=0.8)
for y, v in enumerate(vals):
    ax.text(v+(8 if v > 0 else -8), y, f"{v:+.0f}%", va="center",
            ha="left" if v > 0 else "right", fontsize=8.5)
ax.set_xlabel("薪资相对表现预测值的偏离度（%，基于主要轮换回归模型）")
ax.set_xlim(-140, 560); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig4_residuals.png", dpi=160); plt.close()

# 图5 位置对比
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
order = ["PG", "SG", "SF", "PF", "C"]
bp = axes[0].boxplot([df[df["Pos"] == p]["Salary_M"] for p in order],
                     tick_labels=order, patch_artist=True, showfliers=False,
                     medianprops={"color": "#333"})
for patch in bp["boxes"]: patch.set_facecolor("#B4C6E7")
axes[0].set_ylabel("薪资（百万美元）"); axes[0].set_title("各位置薪资分布", fontsize=10)
bp2 = axes[1].boxplot([df[df["Pos"] == p]["WinScore_48"] for p in order],
                      tick_labels=order, patch_artist=True, showfliers=False,
                      medianprops={"color": "#333"})
for patch in bp2["boxes"]: patch.set_facecolor("#F8CBAD")
axes[1].set_ylabel("Win Score / 48分钟"); axes[1].set_title("各位置单位时间效率", fontsize=10)
for a in axes: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig5_position.png", dpi=160); plt.close()

# 图6 顶薪错位
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.scatter(df[df["Salary"] <= 4e7]["WinScore_48"], df[df["Salary"] <= 4e7]["Salary_M"],
           s=14, alpha=0.4, color=GRAY, label="其余球员")
ax.scatter(max_p["WinScore_48"], max_p["Salary_M"], s=30, color=RED, label="顶薪档(>4000万)")
for _, r in max_p.iterrows():
    if r["Player"] in ["Ben Simmons", "Nikola Jokic", "Rudy Gobert", "Stephen Curry", "Bradley Beal"]:
        ax.annotate(r["Player"], (r["WinScore_48"], r["Salary_M"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
ax.set_xlabel("Win Score / 48分钟"); ax.set_ylabel("薪资（百万美元）")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig6_maxsalary.png", dpi=160); plt.close()

print("\n[完成] 6张图表已输出：fig1~fig6 png")
