"""
MathorCup 2026 C题 - 问题1 & 问题2 完整分析
==============================================
Problem 1A: Key indicators for Phlegm-Dampness (痰湿质) severity
Problem 1B: Key indicators for Hyperlipidemia risk prediction
Problem 2 : Nine-constitution contribution to hyperlipidemia risk
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

import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                             roc_curve, r2_score, mean_squared_error)
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
import shap

# ── Global style ──────────────────────────────────────────────────────────────
rcParams["font.family"] = "DejaVu Sans"   # closest to Arial in most Linux envs
rcParams["axes.titlesize"] = 13
rcParams["axes.labelsize"] = 11
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["legend.fontsize"] = 9

OUT_DIR = "/home/runner/work/Mathorcup/Mathorcup/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0. Load & clean data
# ══════════════════════════════════════════════════════════════════════════════
RAW = "/home/runner/work/Mathorcup/Mathorcup/附件1：样例数据.xlsx"
df_raw = pd.read_excel(RAW)

# English column map
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

# Feature groups
BLOOD_LIPID = ["TC", "TG", "LDL_C", "HDL_C"]
ASSOC_VARS  = ["FastingGlucose", "UricAcid", "BMI"]
ADL_ITEMS   = ["ADL_Toilet","ADL_Eating","ADL_Walking","ADL_Dressing","ADL_Bathing"]
IADL_ITEMS  = ["IADL_Shopping","IADL_Cooking","IADL_Finance","IADL_Transport","IADL_Medication"]
ADL_SCORES  = ["ADL_Total","IADL_Total","Activity_Total"]
DEMO_VARS   = ["AgeGroup","Sex","Smoking","Alcohol"]
CONSTITUTION_SCORES = ["Balanced","QiDeficiency","YangDeficiency","YinDeficiency",
                        "PhlegmDampness","DampHeat","BloodStasis","QiStagnation","SpecialIntrinsic"]
CONSTITUTION_NAMES  = ["Balanced","QiDeficiency","YangDeficiency","YinDeficiency",
                        "PhlegmDampness","DampHeat","BloodStasis","QiStagnation","SpecialIntrinsic"]

# Feature pool for screening (blood lipids + activity scores + associated)
SCREEN_FEATURES = BLOOD_LIPID + ASSOC_VARS + ADL_ITEMS + IADL_ITEMS + ADL_SCORES

print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Hyperlipidemia prevalence: {df['Hyperlipidemia'].mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Baseline characteristics by hyperlipidemia status
# ══════════════════════════════════════════════════════════════════════════════
def baseline_table(df, cat_vars, cont_vars, group_var="Hyperlipidemia"):
    rows = []
    g0 = df[df[group_var] == 0]
    g1 = df[df[group_var] == 1]
    total = df

    for v in cont_vars:
        m_all = f"{total[v].mean():.2f}±{total[v].std():.2f}"
        m0 = f"{g0[v].mean():.2f}±{g0[v].std():.2f}"
        m1 = f"{g1[v].mean():.2f}±{g1[v].std():.2f}"
        _, p = stats.mannwhitneyu(g0[v].dropna(), g1[v].dropna(), alternative="two-sided")
        rows.append({"Variable": v, "Total": m_all,
                     "No Hyperlipidemia (n=%d)"%len(g0): m0,
                     "Hyperlipidemia (n=%d)"%len(g1): m1,
                     "P-value": f"{p:.4f}"})
    for v in cat_vars:
        for lv in sorted(df[v].dropna().unique()):
            n_all = (total[v]==lv).sum()
            n0 = (g0[v]==lv).sum()
            n1 = (g1[v]==lv).sum()
            rows.append({"Variable": f"{v}={lv}",
                         "Total": f"{n_all} ({n_all/len(total)*100:.1f}%)",
                         "No Hyperlipidemia (n=%d)"%len(g0): f"{n0} ({n0/len(g0)*100:.1f}%)" if len(g0)>0 else "—",
                         "Hyperlipidemia (n=%d)"%len(g1): f"{n1} ({n1/len(g1)*100:.1f}%)" if len(g1)>0 else "—",
                         "P-value": ""})
    return pd.DataFrame(rows)

cont_baseline = BLOOD_LIPID + ASSOC_VARS + ADL_SCORES + ["PhlegmDampness"]
cat_baseline  = DEMO_VARS + ["AgeGroup","ConstitutionLabel"]

tbl1 = baseline_table(df, cat_vars=cat_baseline, cont_vars=cont_baseline)
tbl1.to_csv(f"{OUT_DIR}/Table1_Baseline.csv", index=False)
print("✓ Table 1 saved")

# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 1A: Key indicators for Phlegm-Dampness severity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PROBLEM 1A: Phlegm-Dampness Severity Modeling")
print("="*60)

TARGET_A = "PhlegmDampness"
# Use blood lipid + activity + associated variables (exclude all constitution scores as features)
X_A_cols = [c for c in SCREEN_FEATURES if c in df.columns]
X_A = df[X_A_cols].copy()
y_A = df[TARGET_A].values

# ── Step 1: Univariate Spearman correlation + FDR ──────────────────────────
print("\nStep 1: Univariate Spearman correlation (FDR corrected)")
spearman_A = []
for col in X_A_cols:
    r, p = stats.spearmanr(X_A[col], y_A, nan_policy="omit")
    spearman_A.append({"Feature": col, "Spearman_r": r, "P_raw": p})

df_spA = pd.DataFrame(spearman_A)
reject, p_adj, _, _ = multipletests(df_spA["P_raw"], method="fdr_bh")
df_spA["P_adj_FDR"] = p_adj
df_spA["Significant_FDR05"] = reject
df_spA_sig = df_spA[df_spA["Significant_FDR05"]].copy()
print(f"  Significant features (FDR<0.05): {df_spA_sig['Feature'].tolist()}")

# ── Step 2: Elastic Net regression ────────────────────────────────────────
print("\nStep 2: Elastic Net regression")
scaler_A = StandardScaler()
X_A_sc = scaler_A.fit_transform(X_A)

cv_rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
en_A = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    cv=cv_rkf, max_iter=5000, random_state=42, n_jobs=-1)
en_A.fit(X_A_sc, y_A)

en_A_coefs = pd.Series(en_A.coef_, index=X_A_cols)
en_A_selected = en_A_coefs[en_A_coefs != 0].abs().sort_values(ascending=False)
print(f"  Elastic Net l1_ratio={en_A.l1_ratio_:.2f}, alpha={en_A.alpha_:.4f}")
print(f"  Non-zero features: {en_A_selected.index.tolist()}")

# ── Step 3: XGBoost regression ────────────────────────────────────────────
print("\nStep 3: XGBoost regression")
xgb_A = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                           max_depth=4, subsample=0.8,
                           colsample_bytree=0.8, random_state=42,
                           eval_metric="rmse", verbosity=0)
cv5 = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
r2_scores_A = cross_val_score(xgb_A, X_A_sc, y_A, cv=cv5, scoring="r2")
rmse_scores_A = np.sqrt(-cross_val_score(xgb_A, X_A_sc, y_A, cv=cv5, scoring="neg_mean_squared_error"))
print(f"  XGBoost R²={r2_scores_A.mean():.3f}±{r2_scores_A.std():.3f}, RMSE={rmse_scores_A.mean():.3f}±{rmse_scores_A.std():.3f}")

# Final fit for SHAP
xgb_A.fit(X_A_sc, y_A)

# Elastic Net cross-val performance
en_r2 = cross_val_score(en_A, X_A_sc, y_A, cv=cv5, scoring="r2")
en_rmse = np.sqrt(-cross_val_score(en_A, X_A_sc, y_A, cv=cv5, scoring="neg_mean_squared_error"))
print(f"  Elastic Net R²={en_r2.mean():.3f}±{en_r2.std():.3f}, RMSE={en_rmse.mean():.3f}±{en_rmse.std():.3f}")

# Save Table 2
tbl2 = pd.DataFrame({
    "Model": ["Elastic Net", "XGBoost"],
    "R2_mean": [en_r2.mean(), r2_scores_A.mean()],
    "R2_std":  [en_r2.std(),  r2_scores_A.std()],
    "RMSE_mean": [en_rmse.mean(), rmse_scores_A.mean()],
    "RMSE_std":  [en_rmse.std(),  rmse_scores_A.std()],
    "CV": ["5-fold×3", "5-fold×3"]
})
tbl2.to_csv(f"{OUT_DIR}/Table2_PhlegmDampness_ModelPerformance.csv", index=False)
print("✓ Table 2 saved")

# ── SHAP for phlegm-dampness ───────────────────────────────────────────────
explainer_A = shap.TreeExplainer(xgb_A)
shap_vals_A = explainer_A.shap_values(X_A_sc)
df_shap_A = pd.DataFrame({"Feature": X_A_cols,
                            "Mean_SHAP": np.abs(shap_vals_A).mean(axis=0)}).sort_values("Mean_SHAP", ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 1B: Hyperlipidemia risk prediction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PROBLEM 1B: Hyperlipidemia Risk Modeling")
print("="*60)

TARGET_B = "Hyperlipidemia"
# Exclude direct lipid markers (TC/TG/LDL/HDL) from "predictive" screening
# Keep BMI, glucose, uric acid, activity scores as risk markers
X_B_cols_full = SCREEN_FEATURES  # include all for feature selection
X_B = df[X_B_cols_full].copy()
y_B = df[TARGET_B].values

# ── Step 1: Univariate correlation (point-biserial = Spearman for binary) ──
print("\nStep 1: Univariate Spearman / point-biserial (FDR corrected)")
spearman_B = []
for col in X_B_cols_full:
    r, p = stats.spearmanr(X_B[col], y_B, nan_policy="omit")
    spearman_B.append({"Feature": col, "Spearman_r": r, "P_raw": p})

df_spB = pd.DataFrame(spearman_B)
reject_B, p_adj_B, _, _ = multipletests(df_spB["P_raw"], method="fdr_bh")
df_spB["P_adj_FDR"] = p_adj_B
df_spB["Significant_FDR05"] = reject_B
df_spB_sig = df_spB[df_spB["Significant_FDR05"]].copy()
print(f"  Significant features (FDR<0.05): {df_spB_sig['Feature'].tolist()}")

# ── Step 2: Logistic L1 (Lasso) ────────────────────────────────────────────
print("\nStep 2: Logistic Regression (L1) sparse selection")
scaler_B = StandardScaler()
X_B_sc = scaler_B.fit_transform(X_B)

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_l1 = LogisticRegression(penalty="l1", solver="liblinear",
                            C=0.1, class_weight="balanced", max_iter=2000, random_state=42)
# Cross-val AUC
lr_auc = cross_val_score(lr_l1, X_B_sc, y_B, cv=cv_strat, scoring="roc_auc")
lr_f1  = cross_val_score(lr_l1, X_B_sc, y_B, cv=cv_strat, scoring="f1")
print(f"  Logistic L1: AUC={lr_auc.mean():.3f}±{lr_auc.std():.3f}, F1={lr_f1.mean():.3f}±{lr_f1.std():.3f}")
lr_l1.fit(X_B_sc, y_B)
lr_coefs = pd.Series(lr_l1.coef_[0], index=X_B_cols_full)
lr_selected = lr_coefs[lr_coefs != 0].abs().sort_values(ascending=False)
print(f"  L1 non-zero features: {lr_selected.index.tolist()}")

# ── Step 3: XGBoost classifier ────────────────────────────────────────────
print("\nStep 3: XGBoost classifier")
n_neg, n_pos = (y_B==0).sum(), (y_B==1).sum()
scale_pos = n_neg / n_pos
xgb_B = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05,
                            max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, scale_pos_weight=scale_pos,
                            use_label_encoder=False, eval_metric="logloss",
                            random_state=42, verbosity=0)
xgb_auc = cross_val_score(xgb_B, X_B_sc, y_B, cv=cv_strat, scoring="roc_auc")
xgb_f1  = cross_val_score(xgb_B, X_B_sc, y_B, cv=cv_strat, scoring="f1")
print(f"  XGBoost: AUC={xgb_auc.mean():.3f}±{xgb_auc.std():.3f}, F1={xgb_f1.mean():.3f}±{xgb_f1.std():.3f}")
xgb_B.fit(X_B_sc, y_B)

# ── Sensitivity / Specificity at optimal threshold (Youden) ───────────────
def youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx], tpr[idx], 1 - fpr[idx]

# Use cross-validated predicted probabilities for honest evaluation
y_prob_lr  = cross_val_predict(lr_l1,  X_B_sc, y_B, cv=cv_strat, method="predict_proba")[:,1]
y_prob_xgb = cross_val_predict(xgb_B,  X_B_sc, y_B, cv=cv_strat, method="predict_proba")[:,1]
thr_lr, sens_lr, spec_lr = youden_threshold(y_B, y_prob_lr)
thr_xgb, sens_xgb, spec_xgb = youden_threshold(y_B, y_prob_xgb)

# Save Table 3
cv_auc_lr  = roc_auc_score(y_B, y_prob_lr)
cv_auc_xgb = roc_auc_score(y_B, y_prob_xgb)
y_pred_lr_thr  = (y_prob_lr  >= thr_lr).astype(int)
y_pred_xgb_thr = (y_prob_xgb >= thr_xgb).astype(int)
tbl3 = pd.DataFrame({
    "Model": ["Logistic L1", "XGBoost"],
    "AUC_CV": [cv_auc_lr, cv_auc_xgb],
    "F1_CV_mean": [lr_f1.mean(), xgb_f1.mean()],
    "F1_CV_std":  [lr_f1.std(),  xgb_f1.std()],
    "Sensitivity": [sens_lr, sens_xgb],
    "Specificity":  [spec_lr, spec_xgb],
    "CV": ["5-fold Stratified OOF", "5-fold Stratified OOF"]
})
tbl3.to_csv(f"{OUT_DIR}/Table3_Hyperlipidemia_ModelPerformance.csv", index=False)
print("✓ Table 3 saved")

# ── SHAP for hyperlipidemia ────────────────────────────────────────────────
explainer_B = shap.TreeExplainer(xgb_B)
shap_vals_B = explainer_B.shap_values(X_B_sc)

df_shap_B = pd.DataFrame({"Feature": X_B_cols_full,
                            "Mean_SHAP": np.abs(shap_vals_B).mean(axis=0)}).sort_values("Mean_SHAP", ascending=False)

# ── SHAP Stability selection (200 bootstrap repeats) ──────────────────────
print("\nShap Stability selection (100 bootstrap rounds)...")
N_BOOT = 100
freq_A = {c: 0 for c in X_A_cols}
freq_B = {c: 0 for c in X_B_cols_full}

for seed in range(N_BOOT):
    idx = np.random.choice(len(y_A), len(y_A), replace=True)
    Xs_a = X_A_sc[idx]; ys_a = y_A[idx]
    m_a = xgb.XGBRegressor(n_estimators=100, max_depth=3, verbosity=0, random_state=seed)
    m_a.fit(Xs_a, ys_a)
    shap_a = shap.TreeExplainer(m_a).shap_values(Xs_a)
    top_a = np.argsort(np.abs(shap_a).mean(axis=0))[-10:]
    for i in top_a: freq_A[X_A_cols[i]] += 1

    idx2 = np.random.choice(len(y_B), len(y_B), replace=True)
    Xs_b = X_B_sc[idx2]; ys_b = y_B[idx2]
    m_b = xgb.XGBClassifier(n_estimators=100, max_depth=3, verbosity=0,
                              eval_metric="logloss", random_state=seed)
    m_b.fit(Xs_b, ys_b)
    shap_b = shap.TreeExplainer(m_b).shap_values(Xs_b)
    top_b = np.argsort(np.abs(shap_b).mean(axis=0))[-10:]
    for i in top_b: freq_B[X_B_cols_full[i]] += 1

stab_A = pd.Series(freq_A).sort_values(ascending=False) / N_BOOT
stab_B = pd.Series(freq_B).sort_values(ascending=False) / N_BOOT
print(f"  Top-5 stable (1A): {stab_A.head(5).to_dict()}")
print(f"  Top-5 stable (1B): {stab_B.head(5).to_dict()}")

# ── Build Table 4: Final key indicators ───────────────────────────────────
def make_table4(spearman_df, coef_series, shap_mean_df, stab_series, label):
    combined = pd.merge(
        spearman_df[["Feature","Spearman_r","P_adj_FDR","Significant_FDR05"]],
        shap_mean_df[["Feature","Mean_SHAP"]],
        on="Feature", how="left"
    )
    combined["L1_coef"] = combined["Feature"].map(coef_series.to_dict()).fillna(0)
    combined["Stability_Freq"] = combined["Feature"].map(stab_series.to_dict())
    combined["Direction"] = combined["Spearman_r"].apply(lambda r: "Positive" if r>0 else "Negative")
    combined = combined.sort_values("Mean_SHAP", ascending=False)
    combined.to_csv(f"{OUT_DIR}/Table4_KeyIndicators_{label}.csv", index=False)
    return combined

tbl4A = make_table4(df_spA, en_A_coefs, df_shap_A, stab_A, "PhlegmDampness")
tbl4B = make_table4(df_spB, lr_coefs, df_shap_B, stab_B, "Hyperlipidemia")
print("✓ Table 4A & 4B saved")

# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 2: Nine-constitution contribution to hyperlipidemia risk
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PROBLEM 2: Nine-Constitution Contribution to Hyperlipidemia Risk")
print("="*60)

# ── Method A: Multivariate Logistic (OR, 95%CI, P) ────────────────────────
print("\nMethod A: Multivariate Logistic Regression")
confounders = DEMO_VARS + ["BMI","FastingGlucose","UricAcid"]
X_Q2 = df[CONSTITUTION_SCORES + confounders].copy()
X_Q2_sc = pd.DataFrame(StandardScaler().fit_transform(X_Q2),
                        columns=X_Q2.columns)
X_Q2_sm = sm.add_constant(X_Q2_sc)
logit_model = sm.Logit(y_B, X_Q2_sm).fit(disp=0, maxiter=200)
print(logit_model.summary2())

coef_df = pd.DataFrame({
    "Feature": logit_model.params.index,
    "Coef": logit_model.params.values,
    "OR": np.exp(logit_model.params.values),
    "CI_lower": np.exp(logit_model.conf_int()[0].values),
    "CI_upper": np.exp(logit_model.conf_int()[1].values),
    "P_value": logit_model.pvalues.values
})

# ── Method B: XGBoost + SHAP on constitution scores only ──────────────────
print("\nMethod B: XGBoost + SHAP (constitutions + confounders)")
X_Q2_arr = X_Q2_sc.values

xgb_Q2 = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=scale_pos,
                              eval_metric="logloss", verbosity=0, random_state=42)
q2_auc = cross_val_score(xgb_Q2, X_Q2_arr, y_B, cv=cv_strat, scoring="roc_auc")
print(f"  Q2 XGBoost AUC={q2_auc.mean():.3f}±{q2_auc.std():.3f}")
xgb_Q2.fit(X_Q2_arr, y_B)

explainer_Q2 = shap.TreeExplainer(xgb_Q2)
shap_Q2 = explainer_Q2.shap_values(X_Q2_arr)
shap_mean_Q2 = pd.DataFrame({
    "Feature": X_Q2.columns.tolist(),
    "Mean_SHAP": np.abs(shap_Q2).mean(axis=0)
}).sort_values("Mean_SHAP", ascending=False)

# ── Method C: Stratified analysis (sex, age group) ────────────────────────
print("\nMethod C: Stratified SHAP analysis")
CONST_IDX = [X_Q2.columns.tolist().index(c) for c in CONSTITUTION_SCORES]

strat_results = []
for sex_val, sex_label in [(0,"Female"),(1,"Male")]:
    mask = df["Sex"].values == sex_val
    if mask.sum() < 30: continue
    Xsub = X_Q2_arr[mask]; ysub = y_B[mask]
    msub = xgb.XGBClassifier(n_estimators=200, max_depth=3, verbosity=0,
                               eval_metric="logloss", random_state=42)
    msub.fit(Xsub, ysub)
    sh = shap.TreeExplainer(msub).shap_values(Xsub)
    for ci, cname in zip(CONST_IDX, CONSTITUTION_SCORES):
        strat_results.append({"Subgroup": sex_label, "Constitution": cname,
                               "Mean_SHAP": np.abs(sh[:,ci]).mean()})
for age_val in sorted(df["AgeGroup"].unique()):
    mask = df["AgeGroup"].values == age_val
    if mask.sum() < 30: continue
    Xsub = X_Q2_arr[mask]; ysub = y_B[mask]
    msub = xgb.XGBClassifier(n_estimators=200, max_depth=3, verbosity=0,
                               eval_metric="logloss", random_state=42)
    msub.fit(Xsub, ysub)
    sh = shap.TreeExplainer(msub).shap_values(Xsub)
    for ci, cname in zip(CONST_IDX, CONSTITUTION_SCORES):
        strat_results.append({"Subgroup": f"Age{age_val*10+30}-{age_val*10+39}",
                               "Constitution": cname,
                               "Mean_SHAP": np.abs(sh[:,ci]).mean()})

df_strat = pd.DataFrame(strat_results)

# Save Table 5
tbl5 = coef_df[coef_df["Feature"].isin(CONSTITUTION_SCORES)].copy()
tbl5 = tbl5.merge(shap_mean_Q2[["Feature","Mean_SHAP"]],
                  left_on="Feature", right_on="Feature", how="left")
tbl5["RiskRole"] = tbl5["OR"].apply(lambda x: "Risk-Promoting" if x > 1 else "Protective")
tbl5.to_csv(f"{OUT_DIR}/Table5_Constitution_Contribution.csv", index=False)
print("✓ Table 5 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

COLORS = {"primary": "#2E5E9A", "accent": "#E05A42", "green": "#4A9E6B",
          "light": "#AEC6E8", "gray": "#888888", "orange": "#E8952B"}


# ── Figure 1: Feature Importance (Top-10, XGBoost for both tasks) ──────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, shap_df, title, color in zip(
        axes,
        [df_shap_A, df_shap_B],
        ["Phlegm-Dampness Severity\n(XGBoost Feature Importance)",
         "Hyperlipidemia Risk\n(XGBoost Feature Importance)"],
        [COLORS["primary"], COLORS["accent"]]):
    top10 = shap_df.head(10).sort_values("Mean_SHAP")
    bars = ax.barh(top10["Feature"], top10["Mean_SHAP"], color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(title, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, top10["Mean_SHAP"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig1_FeatureImportance.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 1 saved")


# ── Figure 2: SHAP summary plots ──────────────────────────────────────────
# 2A: Phlegm-Dampness
fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_vals_A, X_A_sc, feature_names=X_A_cols,
                  max_display=12, show=False, plot_type="dot",
                  color_bar_label="Feature Value")
plt.title("SHAP Summary: Phlegm-Dampness Severity", fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig2A_SHAP_PhlegmDampness.png", dpi=180, bbox_inches="tight")
plt.close()

# 2B: Hyperlipidemia
fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_vals_B, X_B_sc, feature_names=X_B_cols_full,
                  max_display=12, show=False, plot_type="dot",
                  color_bar_label="Feature Value")
plt.title("SHAP Summary: Hyperlipidemia Risk", fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig2B_SHAP_Hyperlipidemia.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 2A & 2B saved")


# ── Figure 3: Nine-constitution contribution (dual evidence: OR + SHAP) ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Forest plot (OR + 95%CI)
ax = axes[0]
c_sub = coef_df[coef_df["Feature"].isin(CONSTITUTION_SCORES)].copy()
c_sub = c_sub.sort_values("OR", ascending=True)
y_pos = range(len(c_sub))
colors_forest = [COLORS["accent"] if r > 1 else COLORS["green"] for r in c_sub["OR"]]
ax.barh(y_pos, c_sub["OR"] - 1, left=1, color=colors_forest, alpha=0.75, height=0.5)
ax.errorbar(c_sub["OR"], y_pos,
            xerr=[c_sub["OR"] - c_sub["CI_lower"], c_sub["CI_upper"] - c_sub["OR"]],
            fmt="none", color="black", capsize=4, linewidth=1.2)
ax.axvline(x=1, color="black", linestyle="--", linewidth=1)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(c_sub["Feature"], fontsize=9)
ax.set_xlabel("Odds Ratio (OR)")
ax.set_title("Nine Constitutions: OR for\nHyperlipidemia Risk", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
# Add P-value annotation
for i, (_, row) in enumerate(c_sub.iterrows()):
    sig = "***" if row["P_value"] < 0.001 else ("**" if row["P_value"] < 0.01 else
                                                  ("*" if row["P_value"] < 0.05 else ""))
    if sig:
        ax.text(max(row["CI_upper"], row["OR"]) + 0.02, i, sig, va="center",
                color="red", fontsize=9, fontweight="bold")

# Right: SHAP bar chart (constitutions only)
ax = axes[1]
c_shap = shap_mean_Q2[shap_mean_Q2["Feature"].isin(CONSTITUTION_SCORES)].copy()
c_shap = c_shap.sort_values("Mean_SHAP", ascending=True)
colors_shap = [COLORS["primary"] if v > 0 else COLORS["gray"] for v in c_shap["Mean_SHAP"]]
ax.barh(range(len(c_shap)), c_shap["Mean_SHAP"], color=colors_shap, alpha=0.8, height=0.6)
ax.set_yticks(range(len(c_shap)))
ax.set_yticklabels(c_shap["Feature"], fontsize=9)
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title("Nine Constitutions: SHAP Contribution\nto Hyperlipidemia Risk", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

risk_patch = mpatches.Patch(color=COLORS["accent"], alpha=0.75, label="Risk-Promoting (OR>1)")
prot_patch = mpatches.Patch(color=COLORS["green"], alpha=0.75, label="Protective (OR<1)")
axes[0].legend(handles=[risk_patch, prot_patch], loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig3_ConstitutionContribution.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 3 saved")


# ── Figure 4: ROC curves + Confusion Matrix ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC
ax = axes[0]
fpr_lr, tpr_lr, _ = roc_curve(y_B, y_prob_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_B, y_prob_xgb)
auc_lr  = roc_auc_score(y_B, y_prob_lr)
auc_xgb = roc_auc_score(y_B, y_prob_xgb)
ax.plot(fpr_lr, tpr_lr, color=COLORS["primary"], lw=2,
        label=f"Logistic L1 (AUC={auc_lr:.3f})")
ax.plot(fpr_xgb, tpr_xgb, color=COLORS["accent"], lw=2,
        label=f"XGBoost (AUC={auc_xgb:.3f})")
ax.plot([0,1],[0,1],"--", color=COLORS["gray"], lw=1)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curves: Hyperlipidemia Risk Prediction", fontweight="bold")
ax.legend(loc="lower right")
ax.spines[["top","right"]].set_visible(False)

# Confusion Matrix (XGBoost at Youden threshold)
ax = axes[1]
y_pred_xgb = (y_prob_xgb >= thr_xgb).astype(int)
cm = confusion_matrix(y_B, y_pred_xgb)
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i,j] > cm.max()/2 else "black")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Pred: No HLD","Pred: HLD"])
ax.set_yticklabels(["Actual: No HLD","Actual: HLD"])
ax.set_title(f"Confusion Matrix: XGBoost\n(Threshold={thr_xgb:.2f}, "
             f"Sens={sens_xgb:.2f}, Spec={spec_xgb:.2f})", fontweight="bold")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig4_ROC_ConfusionMatrix.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 4 saved")


# ── Figure 5: SHAP Stability barplot (both tasks) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, stab, title, color in zip(
        axes,
        [stab_A.head(12), stab_B.head(12)],
        ["SHAP Stability (Phlegm-Dampness)\n100 Bootstrap Rounds, Top-12",
         "SHAP Stability (Hyperlipidemia Risk)\n100 Bootstrap Rounds, Top-12"],
        [COLORS["primary"], COLORS["accent"]]):
    stab_sorted = stab.sort_values()
    bars = ax.barh(stab_sorted.index, stab_sorted.values, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, label="50% threshold")
    ax.set_xlabel("Selection Frequency")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig5_SHAPStability.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 5 saved")


# ── Figure 6: Stratified SHAP heatmap ────────────────────────────────────
pivot = df_strat.pivot(index="Constitution", columns="Subgroup", values="Mean_SHAP")
fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)), 6))
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f"{pivot.values[i,j]:.3f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="Mean |SHAP|")
ax.set_title("Stratified SHAP: Constitution Contribution\nby Sex and Age Group", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig6_StratifiedSHAP.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 6 saved")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\n[Problem 1A] Phlegm-Dampness Severity")
top5_A = df_shap_A.head(5)
print(f"  Top-5 indicators (SHAP):\n{top5_A.to_string(index=False)}")

print("\n[Problem 1B] Hyperlipidemia Risk")
top5_B = df_shap_B.head(5)
print(f"  Top-5 indicators (SHAP):\n{top5_B.to_string(index=False)}")

print("\n[Problem 2] Nine-Constitution Contribution")
risk_prom = tbl5[tbl5["RiskRole"]=="Risk-Promoting"]["Feature"].tolist()
protective = tbl5[tbl5["RiskRole"]=="Protective"]["Feature"].tolist()
print(f"  Risk-Promoting: {risk_prom}")
print(f"  Protective:     {protective}")

print(f"\n✓ All outputs saved to: {OUT_DIR}/")
print("  Tables: Table1–Table5 (CSV)")
print("  Figures: Fig1–Fig6 (PNG)")
