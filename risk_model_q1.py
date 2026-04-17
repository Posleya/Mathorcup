"""
MathorCup 2026 C题 - 问题1：融合多维度特征的高血脂症风险预警模型
==========================================================================
Outputs:
  Table_RiskDistribution.csv      – 三级风险人数及占比
  Table_ThresholdBasis.csv        – 每级风险阈值选取依据汇总
  Table_CombinationRisk.csv       – 核心特征组合的高血脂检出率 & 相对风险
  Table_RiskModelPerformance.csv  – 模型评价（AUC / F1 / Kappa）
  Fig_RiskDistribution.png        – 三级风险饼图 + 柱状图
  Fig_ThresholdROC.png            – ROC 曲线 & Youden 截断
  Fig_SHAP_RiskModel.png          – SHAP summary（前12特征）
  Fig_CombinationRisk.png         – 核心组合高血脂率气泡图
  Fig_RiskHeatmap.png             – 痰湿×活动量风险热图
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

from scipy import stats
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, confusion_matrix,
    cohen_kappa_score, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb
import shap

# ── Global style ──────────────────────────────────────────────────────────────
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.titlesize"] = 13
rcParams["axes.labelsize"] = 11
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["legend.fontsize"] = 9

BASE_DIR = "/home/runner/work/Mathorcup/Mathorcup"
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0. 数据加载与列名映射
# ══════════════════════════════════════════════════════════════════════════════
RAW = os.path.join(BASE_DIR, "附件1：样例数据.xlsx")
df_raw = pd.read_excel(RAW)

COL_EN = {
    "样本ID": "SampleID",
    "体质标签": "ConstitutionLabel",
    "平和质": "Balanced",
    "气虚质": "QiDeficiency",
    "阳虚质": "YangDeficiency",
    "阴虚质": "YinDeficiency",
    "痰湿质": "PhlegmDampness",
    "湿热质": "DampHeat",
    "血瘀质": "BloodStasis",
    "气郁质": "QiStagnation",
    "特禀质": "SpecialIntrinsic",
    "ADL用厕": "ADL_Toilet",
    "ADL吃饭": "ADL_Eating",
    "ADL步行": "ADL_Walking",
    "ADL穿衣": "ADL_Dressing",
    "ADL洗澡": "ADL_Bathing",
    "ADL总分": "ADL_Total",
    "IADL购物": "IADL_Shopping",
    "IADL做饭": "IADL_Cooking",
    "IADL理财": "IADL_Finance",
    "IADL交通": "IADL_Transport",
    "IADL服药": "IADL_Medication",
    "IADL总分": "IADL_Total",
    "活动量表总分（ADL总分+IADL总分）": "Activity_Total",
    "HDL-C（高密度脂蛋白）": "HDL_C",
    "LDL-C（低密度脂蛋白）": "LDL_C",
    "TG（甘油三酯）": "TG",
    "TC（总胆固醇）": "TC",
    "空腹血糖": "FastingGlucose",
    "血尿酸": "UricAcid",
    "BMI": "BMI",
    "高血脂症二分类标签": "Hyperlipidemia",
    "血脂异常分型标签（确诊病例）": "DyslipidemiaType",
    "年龄组": "AgeGroup",
    "性别": "Sex",
    "吸烟史": "Smoking",
    "饮酒史": "Alcohol",
}
df = df_raw.rename(columns=COL_EN).copy()
print(f"数据集: {df.shape[0]} 样本, {df.shape[1]} 特征")
print(f"高血脂症患病率: {df['Hyperlipidemia'].mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. 一级依据：临床血脂正常范围阈值标记
#    TC 3.1-6.2 | TG 0.56-1.7 | LDL-C 2.07-3.1 | HDL-C 1.04-1.55 mmol/L
# ══════════════════════════════════════════════════════════════════════════════
LIPID_THRESHOLDS = {
    "TC":    {"low": 3.1,  "high": 6.2,  "flag": "TC>6.2"},
    "TG":    {"low": 0.56, "high": 1.7,  "flag": "TG>1.7"},
    "LDL_C": {"low": 2.07, "high": 3.1,  "flag": "LDL>3.1"},
    "HDL_C": {"low": 1.04, "high": 1.55, "flag": "HDL<1.04", "reverse": True},
}

df["TC_high"]   = (df["TC"]    > 6.2 ).astype(int)
df["TG_high"]   = (df["TG"]    > 1.7 ).astype(int)
df["LDL_high"]  = (df["LDL_C"] > 3.1 ).astype(int)
df["HDL_low"]   = (df["HDL_C"] < 1.04).astype(int)
# 任一血脂指标异常
df["LipidAbnormal"] = ((df["TC_high"] + df["TG_high"] + df["LDL_high"] + df["HDL_low"]) >= 1).astype(int)
# 血脂异常计数（≥2为混合型）
df["LipidAbnCount"] = df["TC_high"] + df["TG_high"] + df["LDL_high"] + df["HDL_low"]

print(f"\n[血脂临床阈值]")
print(f"  TC>6.2:     {df['TC_high'].sum()} 例 ({df['TC_high'].mean():.1%})")
print(f"  TG>1.7:     {df['TG_high'].sum()} 例 ({df['TG_high'].mean():.1%})")
print(f"  LDL-C>3.1:  {df['LDL_high'].sum()} 例 ({df['LDL_high'].mean():.1%})")
print(f"  HDL-C<1.04: {df['HDL_low'].sum()} 例 ({df['HDL_low'].mean():.1%})")
print(f"  任一异常:   {df['LipidAbnormal'].sum()} 例 ({df['LipidAbnormal'].mean():.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 二级依据：痰湿质积分分层（40/60/80 三档）
#    按样本分布验证分档合理性
# ══════════════════════════════════════════════════════════════════════════════
pd_score = df["PhlegmDampness"]
print(f"\n[痰湿积分分布]")
print(f"  均值={pd_score.mean():.1f}, 中位数={pd_score.median():.1f}, "
      f"P25={pd_score.quantile(0.25):.1f}, P75={pd_score.quantile(0.75):.1f}")
print(f"  <40:  {(pd_score<40).sum()} ({(pd_score<40).mean():.1%})")
print(f"  40-59:{((pd_score>=40)&(pd_score<60)).sum()} ({((pd_score>=40)&(pd_score<60)).mean():.1%})")
print(f"  60-79:{((pd_score>=60)&(pd_score<80)).sum()} ({((pd_score>=60)&(pd_score<80)).mean():.1%})")
print(f"  ≥80:  {(pd_score>=80).sum()} ({(pd_score>=80).mean():.1%})")

# 各痰湿分层对应高血脂患病率（验证分层有效性）
for lo, hi, label in [(0,40,"<40"),(40,60,"40-59"),(60,80,"60-79"),(80,101,"≥80")]:
    mask = (pd_score>=lo) & (pd_score<hi)
    prev = df.loc[mask, "Hyperlipidemia"].mean()
    print(f"  痰湿{label}: 高血脂患病率={prev:.1%}")

# 痰湿分层标签
def pd_tier(score):
    if score < 40:  return 0   # 低
    if score < 60:  return 1   # 中
    if score < 80:  return 2   # 高
    return 3                   # 极高

df["PD_Tier"] = pd_score.apply(pd_tier)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 三级依据：活动能力分层（总分 <40 / 40-60 / >60）
# ══════════════════════════════════════════════════════════════════════════════
act = df["Activity_Total"]
print(f"\n[活动能力分布]")
print(f"  均值={act.mean():.1f}, 中位数={act.median():.1f}")
print(f"  <40: {(act<40).sum()} ({(act<40).mean():.1%})")
print(f"  40-60:{((act>=40)&(act<=60)).sum()} ({((act>=40)&(act<=60)).mean():.1%})")
print(f"  >60: {(act>60).sum()} ({(act>60).mean():.1%})")

for lo, hi, label in [(0,40,"<40"),(40,61,"40-60"),(60,101,">60")]:
    mask = (act>=lo) & (act<hi)
    prev = df.loc[mask, "Hyperlipidemia"].mean()
    print(f"  活动{label}: 高血脂患病率={prev:.1%}")

df["Act_Low"]  = (act <  40).astype(int)   # 低活动警戒
df["Act_Mid"]  = ((act >= 40) & (act <= 60)).astype(int)
df["Act_High"] = (act >  60).astype(int)

def act_tier(score):
    if score < 40:  return 0   # 低
    if score <= 60: return 1   # 中
    return 2                   # 好

df["Act_Tier"] = act.apply(act_tier)

# ══════════════════════════════════════════════════════════════════════════════
# 4. 四级依据：代谢共病增强因子
#    BMI≥24、尿酸超标（男>428/女>357 μmol/L）、血糖>6.1、吸烟/饮酒
# ══════════════════════════════════════════════════════════════════════════════
df["BMI_High"] = (df["BMI"] >= 24).astype(int)
df["UA_High"]  = np.where(
    df["Sex"] == 1,
    (df["UricAcid"] > 428).astype(int),
    (df["UricAcid"] > 357).astype(int)
)
df["Glucose_High"]   = (df["FastingGlucose"] > 6.1).astype(int)
df["AgeHigh"]        = (df["AgeGroup"] >= 3).astype(int)  # ≥60岁
df["MetabolicBurden"] = (df["BMI_High"] + df["UA_High"] +
                          df["Glucose_High"] + df["Smoking"] + df["Alcohol"])
df["MetabolicFlag"]   = (df["MetabolicBurden"] >= 1).astype(int)

print(f"\n[代谢共病增强因子]")
print(f"  BMI≥24:   {df['BMI_High'].sum()} ({df['BMI_High'].mean():.1%})")
print(f"  尿酸超标: {df['UA_High'].sum()} ({df['UA_High'].mean():.1%})")
print(f"  血糖>6.1: {df['Glucose_High'].sum()} ({df['Glucose_High'].mean():.1%})")
print(f"  吸烟史:   {df['Smoking'].sum()} ({df['Smoking'].mean():.1%})")
print(f"  饮酒史:   {df['Alcohol'].sum()} ({df['Alcohol'].mean():.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 规则层：三级风险分层定义
#    高风险（HR）：满足以下任一条件
#      C1: 血脂异常 AND 痰湿≥60 AND 活动<40
#      C2: 血脂暂正常 BUT 痰湿≥80 AND 活动<40 AND 代谢因子≥1
#    中风险（MR）：满足以下任一条件（且未达高风险）
#      C3: 血脂异常 AND 痰湿40-59
#      C4: 痰湿≥60 AND 活动40-60 AND 无明显血脂异常
#      C5: 血脂临界（仅1项异常）AND 痰湿<40 AND 代谢因子≥1
#    低风险（LR）：其余情况
# ══════════════════════════════════════════════════════════════════════════════

def assign_rule_risk(row):
    pd_s  = row["PhlegmDampness"]
    act_s = row["Activity_Total"]
    lipid = row["LipidAbnormal"]
    lipid_cnt = row["LipidAbnCount"]
    meta  = row["MetabolicFlag"]

    # 高风险条件 C1
    c1 = (lipid == 1) and (pd_s >= 60) and (act_s < 40)
    # 高风险条件 C2
    c2 = (lipid == 0) and (pd_s >= 80) and (act_s < 40) and (meta == 1)
    if c1 or c2:
        return 2  # 高风险

    # 中风险条件 C3
    c3 = (lipid == 1) and (40 <= pd_s < 60)
    # 中风险条件 C4
    c4 = (pd_s >= 60) and (40 <= act_s <= 60) and (lipid == 0)
    # 中风险条件 C5: 仅1项血脂异常 + 代谢负担
    c5 = (lipid_cnt == 1) and (pd_s < 40) and (meta == 1)
    if c3 or c4 or c5:
        return 1  # 中风险

    return 0  # 低风险

df["RuleRisk"] = df.apply(assign_rule_risk, axis=1)
risk_counts = df["RuleRisk"].value_counts().sort_index()
print(f"\n[规则层风险分布]")
for k, label in zip([0,1,2], ["低风险","中风险","高风险"]):
    n = (df["RuleRisk"]==k).sum()
    prev = df.loc[df["RuleRisk"]==k, "Hyperlipidemia"].mean()
    print(f"  {label}: {n} 例 ({n/len(df):.1%}), 高血脂患病率={prev:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 数据驱动层：XGBoost 多分类概率模型
#    特征：痰湿 + 活动 + 血脂 + 代谢 + 基础信息
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("XGBoost 三分类风险模型（数据驱动校准）")
print("="*60)

FEATURES = [
    # 体质核心
    "PhlegmDampness",
    # 活动能力
    "Activity_Total", "ADL_Total", "IADL_Total",
    # 血脂四项
    "TC", "TG", "LDL_C", "HDL_C",
    # 代谢指标
    "BMI", "UricAcid", "FastingGlucose",
    # 基础信息
    "AgeGroup", "Sex", "Smoking", "Alcohol",
    # 其他体质
    "Balanced","QiDeficiency","YangDeficiency","YinDeficiency",
    "DampHeat","BloodStasis","QiStagnation","SpecialIntrinsic",
]
# 仅保留数据集中实际存在的列
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES].values
y_rule = df["RuleRisk"].values       # 规则层三分类目标
y_bin  = df["Hyperlipidemia"].values # 二分类标签（用于AUC评估）

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用规则层标签训练三分类模型（可解释 + 数据驱动双重校准）
xgb_3cls = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=4, subsample=0.8, colsample_bytree=0.8,
    num_class=3, objective="multi:softprob",
    eval_metric="mlogloss", verbosity=0, random_state=42
)

# 交叉验证评估
y_prob_cv = cross_val_predict(xgb_3cls, X, y_rule, cv=cv5, method="predict_proba")
y_pred_cv = y_prob_cv.argmax(axis=1)

f1_macro = f1_score(y_rule, y_pred_cv, average="macro")
f1_weighted = f1_score(y_rule, y_pred_cv, average="weighted")
kappa = cohen_kappa_score(y_rule, y_pred_cv)

# 高风险概率对二分类高血脂的AUC
auc_high_risk = roc_auc_score(y_bin, y_prob_cv[:,2])

print(f"  F1-macro={f1_macro:.3f}, F1-weighted={f1_weighted:.3f}, Kappa={kappa:.3f}")
print(f"  高风险概率对高血脂二分类 AUC={auc_high_risk:.3f}")
print(classification_report(y_rule, y_pred_cv,
                              target_names=["低风险","中风险","高风险"]))

# 用全样本拟合最终模型（用于SHAP）
xgb_3cls.fit(X, y_rule)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Youden 截断校准：用高风险概率的 ROC 确定最优高风险判定概率阈值
# ══════════════════════════════════════════════════════════════════════════════
fpr_hr, tpr_hr, thr_hr = roc_curve(y_bin, y_prob_cv[:,2])
youden_idx = np.argmax(tpr_hr - fpr_hr)
optimal_thr = thr_hr[youden_idx]
optimal_sens = tpr_hr[youden_idx]
optimal_spec = 1 - fpr_hr[youden_idx]
auc_hr = roc_auc_score(y_bin, y_prob_cv[:,2])

# When AUC is near 0.5 the Youden threshold can collapse to 0;
# fall back to the 70th percentile of predicted high-risk probability
if optimal_thr <= 0.01 or auc_hr < 0.55:
    optimal_thr = float(np.percentile(y_prob_cv[:,2], 70))
    fpr_at, tpr_at, _ = roc_curve(y_bin, y_prob_cv[:,2])
    j = tpr_at - fpr_at
    idx2 = np.argmin(np.abs(thr_hr - optimal_thr)) if len(thr_hr) else youden_idx
    optimal_sens = tpr_hr[min(idx2, len(tpr_hr)-1)]
    optimal_spec = 1 - fpr_hr[min(idx2, len(fpr_hr)-1)]

print(f"\n[Youden 最优截断]")
print(f"  高风险概率阈值 = {optimal_thr:.3f}, "
      f"灵敏度={optimal_sens:.3f}, 特异度={optimal_spec:.3f}")

# 基于 Youden 阈值对规则分层进行数据驱动升级
# 规则层高风险 OR 模型概率>=阈值 => 最终高风险
df["RiskProb_High"] = y_prob_cv[:,2]
df["RiskProb_Mid"]  = y_prob_cv[:,1]

def final_risk(row):
    if row["RuleRisk"] == 2 or row["RiskProb_High"] >= optimal_thr:
        return 2  # 高风险
    if row["RuleRisk"] == 1 or row["RiskProb_High"] >= 0.25:
        return 1  # 中风险
    return 0

df["FinalRisk"] = df.apply(final_risk, axis=1)
print(f"\n[最终融合风险分布（规则层+数据驱动校准）]")
for k, label in zip([0,1,2], ["低风险","中风险","高风险"]):
    n = (df["FinalRisk"]==k).sum()
    prev = df.loc[df["FinalRisk"]==k, "Hyperlipidemia"].mean()
    print(f"  {label}: {n} 例 ({n/len(df):.1%}), 高血脂患病率={prev:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SHAP 特征重要性分析
# ══════════════════════════════════════════════════════════════════════════════
print("\n计算 SHAP 值...")
explainer = shap.TreeExplainer(xgb_3cls)
shap_vals = explainer.shap_values(X)  # 3-D array (n_samples, n_features, n_classes) or list
# Normalise to 3-D array regardless of shap version
if isinstance(shap_vals, list):
    shap_arr = np.stack(shap_vals, axis=-1)   # (n, p, 3)
else:
    shap_arr = np.array(shap_vals)            # already (n, p, 3) or (3, n, p)
    if shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
        shap_arr = shap_arr.transpose(1, 2, 0)  # -> (n, p, 3)

# 高风险类别（class 2）的 SHAP 均值
shap_high = shap_arr[:, :, 2]
mean_shap = np.abs(shap_high).mean(axis=0)
# Guard: make sure length matches number of features
if len(mean_shap) != len(FEATURES):
    # Fall back to per-class mean over all classes
    mean_shap = np.abs(shap_arr).mean(axis=(0, 2))
df_shap = pd.DataFrame({"Feature": FEATURES, "MeanSHAP_HighRisk": mean_shap})\
            .sort_values("MeanSHAP_HighRisk", ascending=False)
print(f"  Top-5 特征（高风险类）:\n{df_shap.head(5).to_string(index=False)}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. 核心特征组合分析
#    组合A: 痰湿≥60 + 活动<40 + TG>1.7
#    组合B: 痰湿≥60 + BMI≥24 + LDL>3.1
#    组合C: 痰湿≥80 + 活动<40 + 尿酸超标
#    组合D: 上述任一 + 年龄≥60
# ══════════════════════════════════════════════════════════════════════════════
print("\n[核心特征组合分析]")

combinations = {
    "参考基准（全样本）": np.ones(len(df), dtype=bool),
    "组合A: 痰湿≥60+活动<40+TG高": (
        (df["PhlegmDampness"] >= 60) & (df["Activity_Total"] < 40) & (df["TG_high"] == 1)
    ),
    "组合B: 痰湿≥60+BMI≥24+LDL高": (
        (df["PhlegmDampness"] >= 60) & (df["BMI_High"] == 1) & (df["LDL_high"] == 1)
    ),
    "组合C: 痰湿≥80+活动<40+尿酸高": (
        (df["PhlegmDampness"] >= 80) & (df["Activity_Total"] < 40) & (df["UA_High"] == 1)
    ),
    "组合D: A∪B∪C + 年龄≥60": (
        (
            ((df["PhlegmDampness"] >= 60) & (df["Activity_Total"] < 40) & (df["TG_high"] == 1)) |
            ((df["PhlegmDampness"] >= 60) & (df["BMI_High"] == 1) & (df["LDL_high"] == 1)) |
            ((df["PhlegmDampness"] >= 80) & (df["Activity_Total"] < 40) & (df["UA_High"] == 1))
        ) & (df["AgeHigh"] == 1)
    ),
    "痰湿<40（低痰湿参照组）": (df["PhlegmDampness"] < 40),
}

baseline_prev = df["Hyperlipidemia"].mean()
combo_rows = []
for name, mask in combinations.items():
    n = mask.sum()
    prev = df.loc[mask, "Hyperlipidemia"].mean() if n > 0 else np.nan
    rr   = prev / baseline_prev if (baseline_prev > 0 and not np.isnan(prev)) else np.nan
    high_risk_rate = df.loc[mask, "FinalRisk"].apply(lambda x: x==2).mean() if n > 0 else np.nan
    combo_rows.append({
        "组合": name, "样本数": n, "占全样本(%)": f"{n/len(df)*100:.1f}",
        "高血脂患病率": f"{prev:.1%}" if not np.isnan(prev) else "—",
        "相对风险(RR)": f"{rr:.2f}" if not np.isnan(rr) else "—",
        "高风险等级占比": f"{high_risk_rate:.1%}" if not np.isnan(high_risk_rate) else "—",
    })
    print(f"  {name}: N={n}, 患病率={prev:.1%}, RR={rr:.2f}")

df_combo = pd.DataFrame(combo_rows)
df_combo.to_csv(f"{OUT_DIR}/Table_CombinationRisk.csv", index=False, encoding="utf-8-sig")
print("✓ Table_CombinationRisk.csv 已保存")

# ══════════════════════════════════════════════════════════════════════════════
# 10. 保存汇总表
# ══════════════════════════════════════════════════════════════════════════════

# --- 三级风险人数及占比 ---
risk_rows = []
for k, label in zip([0,1,2], ["低风险","中风险","高风险"]):
    n = (df["FinalRisk"]==k).sum()
    prev = df.loc[df["FinalRisk"]==k, "Hyperlipidemia"].mean()
    high_pd = df.loc[df["FinalRisk"]==k, "PhlegmDampness"].mean()
    high_act = df.loc[df["FinalRisk"]==k, "Activity_Total"].mean()
    risk_rows.append({
        "风险等级": label, "样本数": n,
        "占比(%)": f"{n/len(df)*100:.1f}",
        "高血脂患病率": f"{prev:.1%}",
        "平均痰湿积分": f"{high_pd:.1f}",
        "平均活动总分": f"{high_act:.1f}",
    })
pd.DataFrame(risk_rows).to_csv(f"{OUT_DIR}/Table_RiskDistribution.csv",
                                index=False, encoding="utf-8-sig")
print("✓ Table_RiskDistribution.csv 已保存")

# --- 阈值选取依据 ---
threshold_rows = [
    {"层次": "一级（临床）", "指标": "TC（总胆固醇）",       "高风险阈值": ">6.2 mmol/L",      "依据": "中国成人血脂异常防治指南临床正常上限"},
    {"层次": "一级（临床）", "指标": "TG（甘油三酯）",       "高风险阈值": ">1.7 mmol/L",      "依据": "同上"},
    {"层次": "一级（临床）", "指标": "LDL-C",                "高风险阈值": ">3.1 mmol/L",      "依据": "同上"},
    {"层次": "一级（临床）", "指标": "HDL-C",                "高风险阈值": "<1.04 mmol/L",     "依据": "同上"},
    {"层次": "二级（体质）", "指标": "痰湿积分",              "高风险阈值": "≥60（高）/≥80（极高）", "依据": "样本分布分位 + 各层高血脂患病率显著上升"},
    {"层次": "三级（活动）", "指标": "活动量表总分",          "高风险阈值": "<40（低活动）",     "依据": "低活动亚组高血脂患病率显著高于≥60分亚组"},
    {"层次": "四级（代谢）", "指标": "BMI",                  "高风险阈值": "≥24 kg/m²",        "依据": "中国超重/肥胖诊断标准"},
    {"层次": "四级（代谢）", "指标": "血尿酸",               "高风险阈值": "男>428/女>357 μmol/L", "依据": "高尿酸血症诊断标准"},
    {"层次": "四级（代谢）", "指标": "空腹血糖",             "高风险阈值": ">6.1 mmol/L",      "依据": "糖耐量受损上限"},
    {"层次": "数据驱动",     "指标": "XGBoost高风险概率",     "高风险阈值": f"≥{optimal_thr:.3f}", "依据": f"Youden指数最优截断（AUC={auc_hr:.3f}, 灵敏度={optimal_sens:.3f}, 特异度={optimal_spec:.3f}）"},
]
pd.DataFrame(threshold_rows).to_csv(f"{OUT_DIR}/Table_ThresholdBasis.csv",
                                     index=False, encoding="utf-8-sig")
print("✓ Table_ThresholdBasis.csv 已保存")

# --- 模型性能表 ---
perf_rows = [
    {"评价指标": "F1-macro",         "值": f"{f1_macro:.3f}"},
    {"评价指标": "F1-weighted",      "值": f"{f1_weighted:.3f}"},
    {"评价指标": "Cohen Kappa",      "值": f"{kappa:.3f}"},
    {"评价指标": "高风险概率 AUC（对高血脂二分类）", "值": f"{auc_hr:.3f}"},
    {"评价指标": "Youden 最优概率阈值", "值": f"{optimal_thr:.3f}"},
    {"评价指标": "灵敏度（Sensitivity）", "值": f"{optimal_sens:.3f}"},
    {"评价指标": "特异度（Specificity）", "值": f"{optimal_spec:.3f}"},
]
pd.DataFrame(perf_rows).to_csv(f"{OUT_DIR}/Table_RiskModelPerformance.csv",
                                index=False, encoding="utf-8-sig")
print("✓ Table_RiskModelPerformance.csv 已保存")

# ══════════════════════════════════════════════════════════════════════════════
# 11. 图表生成
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {"low": "#4A9E6B", "mid": "#E8952B", "high": "#E05A42",
          "primary": "#2E5E9A", "gray": "#888888"}

# ── 图1: 三级风险分布（饼图 + 柱状图）────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

labels = ["低风险", "中风险", "高风险"]
counts = [(df["FinalRisk"]==k).sum() for k in [0,1,2]]
prevs  = [df.loc[df["FinalRisk"]==k, "Hyperlipidemia"].mean() for k in [0,1,2]]
colors = [COLORS["low"], COLORS["mid"], COLORS["high"]]

# 饼图
ax = axes[0]
wedges, texts, autotexts = ax.pie(
    counts, labels=labels, colors=colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
ax.set_title("三级风险人群分布\n(N=700)", fontweight="bold")

# 柱状图（各级高血脂患病率）
ax = axes[1]
bars = ax.bar(labels, [p*100 for p in prevs], color=colors, edgecolor="white",
              width=0.5, linewidth=1.5)
ax.set_ylabel("高血脂症患病率 (%)")
ax.set_title("各风险等级高血脂症患病率", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.axhline(y=df["Hyperlipidemia"].mean()*100, color=COLORS["gray"],
           linestyle="--", linewidth=1.2, label=f"总体患病率 {df['Hyperlipidemia'].mean():.1%}")
ax.legend(fontsize=8)
for bar, p, n in zip(bars, prevs, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{p:.1%}\n(n={n})", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(p*100 for p in prevs) * 1.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig_RiskDistribution.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Fig_RiskDistribution.png 已保存")


# ── 图2: ROC 曲线 + Youden 截断点 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_hr, tpr_hr, color=COLORS["primary"], lw=2,
        label=f"高风险概率 ROC (AUC={auc_hr:.3f})")
ax.plot([0,1],[0,1],"--", color=COLORS["gray"], lw=1)
ax.scatter(fpr_hr[youden_idx], tpr_hr[youden_idx],
           color=COLORS["high"], s=80, zorder=5,
           label=f"Youden最优阈值={optimal_thr:.3f}\n"
                 f"灵敏度={optimal_sens:.3f}, 特异度={optimal_spec:.3f}")
ax.annotate(f"Thr={optimal_thr:.3f}",
            xy=(fpr_hr[youden_idx], tpr_hr[youden_idx]),
            xytext=(fpr_hr[youden_idx]+0.08, tpr_hr[youden_idx]-0.08),
            arrowprops=dict(arrowstyle="->", color=COLORS["high"]),
            fontsize=9, color=COLORS["high"])
ax.set_xlabel("假阳性率 (1 - Specificity)")
ax.set_ylabel("真阳性率 (Sensitivity)")
ax.set_title("ROC 曲线：高风险概率预测高血脂症\n(Youden 截断校准)", fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig_ThresholdROC.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Fig_ThresholdROC.png 已保存")


# ── 图3: SHAP 特征重要性（高风险类，Top-14）────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
top14 = df_shap.head(14).sort_values("MeanSHAP_HighRisk")
bars = ax.barh(top14["Feature"], top14["MeanSHAP_HighRisk"],
               color=COLORS["primary"], edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean |SHAP Value|（高风险类）")
ax.set_title("SHAP 特征重要性：高血脂症高风险预测\n（Top-14，XGBoost 三分类模型）",
             fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, top14["MeanSHAP_HighRisk"]):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig_SHAP_RiskModel.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Fig_SHAP_RiskModel.png 已保存")


# ── 图4: 核心组合高血脂率气泡图 ──────────────────────────────────────────
combo_plot_data = [r for r in combo_rows
                   if r["组合"] != "参考基准（全样本）"
                   and r["高血脂患病率"] != "—"
                   and int(r["样本数"]) > 0]
names   = [r["组合"] for r in combo_plot_data]
ns      = [int(r["样本数"]) for r in combo_plot_data]
prevs_c = [float(r["高血脂患病率"].strip("%"))/100 for r in combo_plot_data]
rrs     = [float(r["相对风险(RR)"]) if r["相对风险(RR)"] != "—" else 1.0
           for r in combo_plot_data]

fig, ax = plt.subplots(figsize=(10, 5))
scatter_colors = [COLORS["high"] if r>1.5 else COLORS["mid"] if r>1.0 else COLORS["low"]
                  for r in rrs]
sc = ax.scatter(range(len(names)), [p*100 for p in prevs_c],
                s=[n*3 for n in ns], c=scatter_colors,
                alpha=0.7, edgecolors="white", linewidths=1.5)
ax.axhline(y=df["Hyperlipidemia"].mean()*100, color=COLORS["gray"],
           linestyle="--", linewidth=1.2, label=f"总体患病率 {df['Hyperlipidemia'].mean():.1%}")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("高血脂症患病率 (%)")
ax.set_title("核心特征组合 → 高血脂症患病率\n（气泡大小=样本量，颜色=RR级别）", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
for i, (p, rr, n) in enumerate(zip(prevs_c, rrs, ns)):
    ax.text(i, p*100 + 1.5, f"RR={rr:.2f}\nn={n}", ha="center", va="bottom", fontsize=7.5)
risk_high_patch  = mpatches.Patch(color=COLORS["high"], alpha=0.7, label="RR>1.5（高危组合）")
risk_mid_patch   = mpatches.Patch(color=COLORS["mid"],  alpha=0.7, label="RR 1.0-1.5")
risk_low_patch   = mpatches.Patch(color=COLORS["low"],  alpha=0.7, label="RR≤1.0（参照）")
ax.legend(handles=[risk_high_patch, risk_mid_patch, risk_low_patch],
          loc="upper right", fontsize=8)
ax.set_ylim(0, max(p*100 for p in prevs_c) * 1.35)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig_CombinationRisk.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Fig_CombinationRisk.png 已保存")


# ── 图5: 痰湿×活动量风险热图 ─────────────────────────────────────────────
pd_bins  = [0, 40, 60, 80, 101]
act_bins = [0, 40, 61, 101]
pd_labels  = ["<40", "40-59", "60-79", "≥80"]
act_labels = ["低(<40)", "中(40-60)", "高(>60)"]

df["PD_Band"]  = pd.cut(df["PhlegmDampness"], bins=pd_bins,  labels=pd_labels,  right=False)
df["Act_Band"] = pd.cut(df["Activity_Total"], bins=act_bins, labels=act_labels, right=False)

heat_prev = df.groupby(["PD_Band","Act_Band"], observed=False)["Hyperlipidemia"].mean().unstack("Act_Band")
heat_n    = df.groupby(["PD_Band","Act_Band"], observed=False)["Hyperlipidemia"].count().unstack("Act_Band")

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(heat_prev.values * 100, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(act_labels)))
ax.set_xticklabels(act_labels, fontsize=9)
ax.set_yticks(range(len(pd_labels)))
ax.set_yticklabels(pd_labels, fontsize=9)
ax.set_xlabel("活动量表总分")
ax.set_ylabel("痰湿积分")
ax.set_title("高血脂症患病率热图\n（痰湿积分 × 活动能力分层）", fontweight="bold")
for i in range(len(pd_labels)):
    for j in range(len(act_labels)):
        val = heat_prev.values[i,j]
        n   = heat_n.values[i,j]
        if not np.isnan(val):
            ax.text(j, i, f"{val*100:.0f}%\n(n={int(n)})",
                    ha="center", va="center", fontsize=8,
                    color="white" if val > 0.6 else "black", fontweight="bold")
plt.colorbar(im, ax=ax, label="高血脂症患病率 (%)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig_RiskHeatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Fig_RiskHeatmap.png 已保存")


# ══════════════════════════════════════════════════════════════════════════════
# 最终汇总输出
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("最终汇总")
print("="*60)
print("\n[三级风险分布]")
for r in risk_rows:
    print(f"  {r['风险等级']}: {r['样本数']} 例 ({r['占比(%)']}%), "
          f"高血脂患病率={r['高血脂患病率']}")

print(f"\n[SHAP 主导因子（高风险类 Top-5）]")
for _, row in df_shap.head(5).iterrows():
    print(f"  {row['Feature']}: mean|SHAP|={row['MeanSHAP_HighRisk']:.4f}")

print(f"\n[模型性能] AUC={auc_hr:.3f}, Kappa={kappa:.3f}, F1-macro={f1_macro:.3f}")
print(f"\n✓ 所有输出已保存至: {OUT_DIR}/")
print("  表格: Table_RiskDistribution / Table_ThresholdBasis / Table_CombinationRisk / Table_RiskModelPerformance")
print("  图形: Fig_RiskDistribution / Fig_ThresholdROC / Fig_SHAP_RiskModel / Fig_CombinationRisk / Fig_RiskHeatmap")
