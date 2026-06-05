"""
╔══════════════════════════════════════════════════════════════════╗
║     ONLINE PAYMENT FRAUD DETECTION USING MACHINE LEARNING        ║
║     Libraries: Pandas | NumPy | Matplotlib | Seaborn | Sklearn   ║
╚══════════════════════════════════════════════════════════════════╝
"""
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  GLOBAL STYLE CONFIGURATION
# ─────────────────────────────────────────────
DARK_BG    = "#0A0E1A"
CARD_BG    = "#111827"
ACCENT1    = "#00F5C4"   # cyan-green
ACCENT2    = "#FF3CAC"   # magenta
ACCENT3    = "#F7B731"   # amber
ACCENT4    = "#5B8AF5"   # soft blue
SAFE_CLR   = "#00E676"
FRAUD_CLR  = "#FF1744"
TEXT_LIGHT = "#E8EAF6"
TEXT_DIM   = "#7986CB"

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    CARD_BG,
    'axes.edgecolor':    '#1E2A3A',
    'axes.labelcolor':   TEXT_LIGHT,
    'xtick.color':       TEXT_DIM,
    'ytick.color':       TEXT_DIM,
    'text.color':        TEXT_LIGHT,
    'grid.color':        '#1E2A3A',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
    'legend.facecolor':  CARD_BG,
    'legend.edgecolor':  '#1E2A3A',
})

FRAUD_CMAP   = LinearSegmentedColormap.from_list("fraud", [DARK_BG, ACCENT4, ACCENT2])
SAFE_CMAP    = LinearSegmentedColormap.from_list("safe",  [DARK_BG, ACCENT4, ACCENT1])


# ══════════════════════════════════════════════
#  1.  DATA GENERATION
# ══════════════════════════════════════════════
np.random.seed(42)
N = 10_000

def generate_dataset(n):
    tx_types   = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    tx_weights = [0.35, 0.25, 0.20, 0.12, 0.08]

    tx_type    = np.random.choice(tx_types, n, p=tx_weights)
    amount     = np.where(
        np.random.rand(n) < 0.05,
        np.random.exponential(50_000, n),        # large outlier amounts
        np.random.lognormal(7, 1.5, n)
    )

    old_bal_orig  = np.random.uniform(0, 500_000, n)
    new_bal_orig  = np.maximum(0, old_bal_orig - amount * np.random.uniform(0.5, 1.2, n))
    old_bal_dest  = np.random.uniform(0, 300_000, n)

    hour_of_day   = np.random.randint(0, 24, n)
    day_of_week   = np.random.randint(0, 7, n)
    step          = np.random.randint(1, 744, n)   # hours in a month

    # Fraud logic: higher prob for TRANSFER/CASH_OUT + large amounts + odd hours
    fraud_prob = (
        0.005
        + 0.08 * np.isin(tx_type, ['TRANSFER', 'CASH_OUT']).astype(float)
        + 0.05 * (amount > 200_000).astype(float)
        + 0.03 * ((hour_of_day < 5) | (hour_of_day > 22)).astype(float)
        + 0.02 * (old_bal_orig < 1000).astype(float)
    )
    fraud_prob = np.clip(fraud_prob, 0, 0.95)
    is_fraud   = (np.random.rand(n) < fraud_prob).astype(int)

    # New dest balance (fraudsters drain accounts)
    new_bal_dest = np.where(
        is_fraud == 1,
        old_bal_dest + amount,
        old_bal_dest + amount * np.random.uniform(0.0, 0.5, n)
    )

    df = pd.DataFrame({
        'step':           step,
        'type':           tx_type,
        'amount':         amount,
        'oldbalanceOrg':  old_bal_orig,
        'newbalanceOrig': new_bal_orig,
        'oldbalanceDest': old_bal_dest,
        'newbalanceDest': new_bal_dest,
        'hour_of_day':    hour_of_day,
        'day_of_week':    day_of_week,
        'isFraud':        is_fraud,
    })
    return df

df = generate_dataset(N)
print(f"✔  Dataset created  →  {df.shape[0]:,} rows  |  {df.shape[1]} columns")
print(f"   Fraud rate: {df['isFraud'].mean()*100:.2f}%")


# ══════════════════════════════════════════════
#  2.  FEATURE ENGINEERING
# ══════════════════════════════════════════════
le = LabelEncoder()
df['type_enc'] = le.fit_transform(df['type'])

df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['amount_to_balance']  = df['amount'] / (df['oldbalanceOrg'] + 1)
df['is_night_tx']        = ((df['hour_of_day'] < 5) | (df['hour_of_day'] > 22)).astype(int)
df['is_large_tx']        = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
df['balance_zero_orig']  = (df['newbalanceOrig'] == 0).astype(int)

FEATURES = [
    'step', 'type_enc', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'hour_of_day', 'day_of_week',
    'balance_diff_orig', 'balance_diff_dest', 'amount_to_balance',
    'is_night_tx', 'is_large_tx', 'balance_zero_orig'
]
X = df[FEATURES]
y = df['isFraud']

# Balance classes via upsampling
df_maj  = df[df['isFraud'] == 0]
df_min  = df[df['isFraud'] == 1]
df_min_up = resample(df_min, replace=True, n_samples=len(df_maj)//3, random_state=42)
df_bal  = pd.concat([df_maj, df_min_up])
X_bal   = df_bal[FEATURES]
y_bal   = df_bal['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.25, stratify=y_bal, random_state=42
)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"✔  Features engineered  →  {len(FEATURES)} features")
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ══════════════════════════════════════════════
#  3.  MODEL TRAINING
# ══════════════════════════════════════════════
models = {
    'Random Forest':       RandomForestClassifier(n_estimators=120, max_depth=12,
                                                   class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                       learning_rate=0.08, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced',
                                               C=0.5, random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name == 'Logistic Regression' else X_train
    Xte = X_test_sc  if name == 'Logistic Regression' else X_test
    model.fit(Xtr, y_train)
    y_prob = model.predict_proba(Xte)[:, 1]
    y_pred = model.predict(Xte)
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'model': model, 'y_prob': y_prob, 'y_pred': y_pred,
        'auc': auc, 'ap': ap, 'report': report,
        'fpr': None, 'tpr': None, 'prec': None, 'rec': None,
    }
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    results[name]['fpr']  = fpr
    results[name]['tpr']  = tpr
    results[name]['prec'] = prec
    results[name]['rec']  = rec
    print(f"✔  {name:<22}  AUC={auc:.4f}  AP={ap:.4f}")

best_name   = max(results, key=lambda k: results[k]['auc'])
best_result = results[best_name]
print(f"\n🏆  Best model: {best_name}  (AUC={best_result['auc']:.4f})")


# ══════════════════════════════════════════════
#  4.  VISUALIZATION SUITE  (6 figures)
# ══════════════════════════════════════════════

# ── FIG 1 ── EDA Dashboard ─────────────────────────────────────────
fig1 = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
fig1.suptitle("ONLINE PAYMENT FRAUD  ·  Exploratory Data Analysis",
              fontsize=22, fontweight='bold', color=ACCENT1,
              y=0.97)

gs = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.38,
                       top=0.93, bottom=0.06, left=0.07, right=0.97)

# 4.1  Fraud vs Legit donut
ax_donut = fig1.add_subplot(gs[0, 0])
counts   = df['isFraud'].value_counts()
wedges, texts, autotexts = ax_donut.pie(
    counts, autopct='%1.1f%%', pctdistance=0.78, startangle=90,
    colors=[SAFE_CLR, FRAUD_CLR],
    wedgeprops=dict(width=0.5, edgecolor=DARK_BG, linewidth=3),
)
for at in autotexts:
    at.set(fontsize=13, fontweight='bold', color=DARK_BG)
ax_donut.set_title("Transaction Class Split", color=TEXT_LIGHT, fontsize=13, pad=14)
legend_patches = [
    mpatches.Patch(color=SAFE_CLR,  label=f"Legitimate  ({counts[0]:,})"),
    mpatches.Patch(color=FRAUD_CLR, label=f"Fraudulent  ({counts[1]:,})"),
]
ax_donut.legend(handles=legend_patches, loc='lower center',
                bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=9)

# 4.2  Fraud rate by transaction type
ax_type = fig1.add_subplot(gs[0, 1])
fraud_by_type = df.groupby('type')['isFraud'].mean().sort_values(ascending=False) * 100
bars = ax_type.barh(fraud_by_type.index, fraud_by_type.values,
                    color=[FRAUD_CLR if v > 5 else ACCENT4 for v in fraud_by_type.values],
                    edgecolor=DARK_BG, linewidth=0.8, height=0.65)
for bar, val in zip(bars, fraud_by_type.values):
    ax_type.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va='center', fontsize=10, color=TEXT_LIGHT)
ax_type.set_title("Fraud Rate by TX Type", color=TEXT_LIGHT, fontsize=13)
ax_type.set_xlabel("Fraud Rate (%)")
ax_type.axvline(fraud_by_type.mean(), color=ACCENT3, ls='--', lw=1.2, alpha=0.7)
ax_type.grid(axis='x', alpha=0.3)

# 4.3  Amount distribution (log scale)
ax_amt = fig1.add_subplot(gs[0, 2])
for label, color, alpha in [(0, SAFE_CLR, 0.55), (1, FRAUD_CLR, 0.80)]:
    data = np.log1p(df[df['isFraud'] == label]['amount'])
    ax_amt.hist(data, bins=50, color=color, alpha=alpha,
                label='Legitimate' if label==0 else 'Fraud', edgecolor='none')
ax_amt.set_title("Transaction Amount  (log₁₊ₓ)", color=TEXT_LIGHT, fontsize=13)
ax_amt.set_xlabel("log(Amount + 1)")
ax_amt.set_ylabel("Count")
ax_amt.legend(fontsize=9)
ax_amt.grid(alpha=0.3)

# 4.4  Hourly fraud heatmap
ax_heat = fig1.add_subplot(gs[1, :2])
pivot = df.pivot_table(index='day_of_week', columns='hour_of_day',
                       values='isFraud', aggfunc='mean') * 100
days  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sns.heatmap(pivot, ax=ax_heat, cmap='RdYlGn_r', linewidths=0.4,
            linecolor='#0A0E1A', annot=False,
            cbar_kws={'label': 'Fraud Rate (%)', 'shrink': 0.8})
ax_heat.set_title("Fraud Rate Heatmap  ·  Day × Hour", color=TEXT_LIGHT, fontsize=13)
ax_heat.set_yticklabels(days, rotation=0, fontsize=9)
ax_heat.set_xlabel("Hour of Day")
ax_heat.set_ylabel("Day of Week")
ax_heat.tick_params(axis='x', labelsize=8)

# 4.5  Balance diff violin
ax_viol = fig1.add_subplot(gs[1, 2])
viol_data = [
    np.clip(df[df['isFraud'] == 0]['balance_diff_orig'].values, -1e5, 1e5),
    np.clip(df[df['isFraud'] == 1]['balance_diff_orig'].values, -1e5, 1e5),
]
vp = ax_viol.violinplot(viol_data, positions=[1, 2],
                         showmedians=True, showextrema=False)
colors_v = [SAFE_CLR, FRAUD_CLR]
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(colors_v[i])
    body.set_alpha(0.7)
vp['cmedians'].set_color(ACCENT3)
ax_viol.set_xticks([1, 2])
ax_viol.set_xticklabels(['Legitimate', 'Fraud'], fontsize=10)
ax_viol.set_title("Balance Δ Distribution", color=TEXT_LIGHT, fontsize=13)
ax_viol.set_ylabel("Balance Change (clipped)")
ax_viol.axhline(0, color=TEXT_DIM, ls='--', lw=1, alpha=0.6)

# 4.6  Correlation heatmap (numeric features)
ax_corr = fig1.add_subplot(gs[2, :])
num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
            'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest',
            'amount_to_balance', 'hour_of_day', 'isFraud']
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
custom_cmap = LinearSegmentedColormap.from_list("corr", [FRAUD_CLR, DARK_BG, ACCENT1])
sns.heatmap(corr_matrix, ax=ax_corr, cmap=custom_cmap, mask=mask,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            linewidths=0.5, linecolor='#0A0E1A', square=True,
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.6})
ax_corr.set_title("Feature Correlation Matrix", color=TEXT_LIGHT, fontsize=13)
ax_corr.tick_params(axis='both', labelsize=8)
plt.setp(ax_corr.get_xticklabels(), rotation=30, ha='right')

fig1.savefig('outputs/01_EDA_dashboard.png',
             dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("✔  Figure 1 saved → 01_EDA_dashboard.png")


# ── FIG 2 ── Model Performance: ROC + PR curves ────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK_BG)
fig2.suptitle("MODEL PERFORMANCE  ·  ROC & Precision-Recall Curves",
              fontsize=18, fontweight='bold', color=ACCENT1, y=1.02)

palette = [ACCENT1, ACCENT2, ACCENT3]

ax_roc = axes2[0]
ax_roc.plot([0, 1], [0, 1], color=TEXT_DIM, lw=1.2, ls='--', label='Random (AUC=0.50)')
for (name, res), col in zip(results.items(), palette):
    ax_roc.plot(res['fpr'], res['tpr'], color=col, lw=2.5,
                label=f"{name}  (AUC={res['auc']:.4f})")
ax_roc.fill_between(results[best_name]['fpr'], results[best_name]['tpr'],
                    alpha=0.08, color=ACCENT1)
ax_roc.set(title='ROC Curve', xlabel='False Positive Rate', ylabel='True Positive Rate')
ax_roc.legend(fontsize=10, loc='lower right')
ax_roc.grid(alpha=0.3)

ax_pr = axes2[1]
for (name, res), col in zip(results.items(), palette):
    ax_pr.plot(res['rec'], res['prec'], color=col, lw=2.5,
               label=f"{name}  (AP={res['ap']:.4f})")
ax_pr.fill_between(results[best_name]['rec'], results[best_name]['prec'],
                   alpha=0.08, color=ACCENT2)
baseline = y_test.mean()
ax_pr.axhline(baseline, color=TEXT_DIM, ls='--', lw=1.2,
              label=f'Baseline  (AP={baseline:.4f})')
ax_pr.set(title='Precision-Recall Curve', xlabel='Recall', ylabel='Precision')
ax_pr.legend(fontsize=10)
ax_pr.grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig('outputs/02_model_curves.png',
             dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("✔  Figure 2 saved → 02_model_curves.png")


# ── FIG 3 ── Confusion Matrix Triptych ────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6), facecolor=DARK_BG)
fig3.suptitle("CONFUSION MATRICES  ·  All Models", fontsize=18,
              fontweight='bold', color=ACCENT1, y=1.02)

for ax, (name, res) in zip(axes3, results.items()):
    cm     = confusion_matrix(y_test, res['y_pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cmap_cm = LinearSegmentedColormap.from_list("cm", [CARD_BG, ACCENT4])
    im = ax.imshow(cm_pct, cmap=cmap_cm, vmin=0, vmax=100)
    labels = [['True Neg', 'False Pos'], ['False Neg', 'True Pos']]
    clrs   = [[SAFE_CLR, FRAUD_CLR], [FRAUD_CLR, SAFE_CLR]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i,j]:.1f}%)",
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color=clrs[i][j])
            ax.text(j, i + 0.38, labels[i][j], ha='center', va='center',
                    fontsize=8, color='#90A4AE', style='italic')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nLegit', 'Predicted\nFraud'], fontsize=10)
    ax.set_yticklabels(['Actual\nLegit', 'Actual\nFraud'], fontsize=10)
    prec1 = res['report']['1']['precision']
    rec1  = res['report']['1']['recall']
    f1    = res['report']['1']['f1-score']
    ax.set_title(f"{name}\nPrec={prec1:.3f}  Rec={rec1:.3f}  F1={f1:.3f}",
                 color=TEXT_LIGHT, fontsize=11, pad=14)

fig3.tight_layout()
fig3.savefig('outputs/03_confusion_matrices.png',
             dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("✔  Figure 3 saved → 03_confusion_matrices.png")


# ── FIG 4 ── Feature Importance (Best Model) ──────────────────────
fig4, ax4 = plt.subplots(figsize=(14, 8), facecolor=DARK_BG)

rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
feat_sorted = [FEATURES[i] for i in sorted_idx]
imp_sorted  = importances[sorted_idx]

bar_colors = [FRAUD_CLR if imp > np.percentile(imp_sorted, 75)
              else ACCENT4 if imp > np.percentile(imp_sorted, 50)
              else TEXT_DIM for imp in imp_sorted]

bars4 = ax4.bar(range(len(feat_sorted)), imp_sorted,
                color=bar_colors, edgecolor=DARK_BG, linewidth=0.8)

for i, (bar, imp) in enumerate(zip(bars4, imp_sorted)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{imp:.3f}", ha='center', va='bottom', fontsize=8,
             color=TEXT_LIGHT, rotation=45)

ax4.set_xticks(range(len(feat_sorted)))
ax4.set_xticklabels(feat_sorted, rotation=45, ha='right', fontsize=10)
ax4.set_title("FEATURE IMPORTANCE  ·  Random Forest (Best Model)",
              fontsize=16, fontweight='bold', color=ACCENT1, pad=18)
ax4.set_ylabel("Importance Score", fontsize=12)
ax4.grid(axis='y', alpha=0.3)

legend_elems = [
    mpatches.Patch(color=FRAUD_CLR,  label='Top 25% — Critical Features'),
    mpatches.Patch(color=ACCENT4,    label='Top 50% — Important Features'),
    mpatches.Patch(color=TEXT_DIM,   label='Supporting Features'),
]
ax4.legend(handles=legend_elems, loc='upper right', fontsize=10)

fig4.tight_layout()
fig4.savefig('outputs/04_feature_importance.png',
             dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("✔  Figure 4 saved → 04_feature_importance.png")


# ── FIG 5 ── Score Distribution + Threshold Analysis ──────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK_BG)
fig5.suptitle("FRAUD SCORE ANALYSIS  ·  Probability Distribution & Threshold Impact",
              fontsize=17, fontweight='bold', color=ACCENT1, y=1.02)

best_probs = best_result['y_prob']

ax5l = axes5[0]
for label, color, alpha, name in [(0, SAFE_CLR, 0.55, 'Legitimate'),
                                    (1, FRAUD_CLR, 0.80, 'Fraud')]:
    mask = y_test == label
    ax5l.hist(best_probs[mask], bins=60, color=color, alpha=alpha,
              label=f"{name} ({mask.sum():,})", edgecolor='none', density=True)
ax5l.axvline(0.5, color=ACCENT3, ls='--', lw=2, label='Default threshold (0.5)')
ax5l.set_title(f"Fraud Probability Distribution\n({best_name})",
               color=TEXT_LIGHT, fontsize=13)
ax5l.set_xlabel("Predicted Fraud Probability")
ax5l.set_ylabel("Density")
ax5l.legend(fontsize=10)
ax5l.grid(alpha=0.3)

ax5r = axes5[1]
thresholds = np.linspace(0.05, 0.95, 60)
precisions, recalls, f1_scores, fprs_t = [], [], [], []
for t in thresholds:
    y_t = (best_probs >= t).astype(int)
    tp = ((y_t == 1) & (y_test == 1)).sum()
    fp = ((y_t == 1) & (y_test == 0)).sum()
    fn = ((y_t == 0) & (y_test == 1)).sum()
    tn = ((y_t == 0) & (y_test == 0)).sum()
    prec_t = tp / (tp + fp + 1e-9)
    rec_t  = tp / (tp + fn + 1e-9)
    f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-9)
    fpr_t  = fp / (fp + tn + 1e-9)
    precisions.append(prec_t); recalls.append(rec_t)
    f1_scores.append(f1_t);   fprs_t.append(fpr_t)

ax5r.plot(thresholds, precisions, color=ACCENT1,  lw=2.5, label='Precision')
ax5r.plot(thresholds, recalls,    color=FRAUD_CLR, lw=2.5, label='Recall')
ax5r.plot(thresholds, f1_scores,  color=ACCENT3,  lw=2.5, label='F1 Score')
ax5r.plot(thresholds, fprs_t,     color=TEXT_DIM,  lw=1.5, ls='--', label='FPR')
best_t = thresholds[np.argmax(f1_scores)]
ax5r.axvline(best_t, color=ACCENT2, ls=':', lw=2,
             label=f'Best F1 threshold = {best_t:.2f}')
ax5r.fill_between(thresholds, f1_scores, alpha=0.08, color=ACCENT3)
ax5r.set_title("Metric vs. Decision Threshold", color=TEXT_LIGHT, fontsize=13)
ax5r.set_xlabel("Threshold")
ax5r.set_ylabel("Score")
ax5r.legend(fontsize=10)
ax5r.grid(alpha=0.3)

fig5.tight_layout()
fig5.savefig('outputs/05_score_analysis.png',
             dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("✔  Figure 5 saved → 05_score_analysis.png")


# ── FIG 6 ── Executive Summary Card ───────────────────────────────
fig6 = plt.figure(figsize=(20, 11), facecolor=DARK_BG)
fig6.suptitle("FRAUD DETECTION  ·  Executive Summary",
              fontsize=22, fontweight='bold', color=ACCENT1, y=0.98)

# Background accent bars
for x, w, col in [(0, 0.015, FRAUD_CLR), (0.985, 0.015, ACCENT1)]:
    fig6.add_axes([x, 0, w, 1]).set_facecolor(col); plt.gca().axis('off')

# ── KPI Cards ──
kpi_ax = fig6.add_axes([0.02, 0.68, 0.96, 0.24])
kpi_ax.axis('off')
kpi_ax.set_facecolor(DARK_BG)

kpis = [
    ("Dataset Size",        f"{N:,}",             "transactions simulated", ACCENT4),
    ("Fraud Rate",          f"{df['isFraud'].mean()*100:.2f}%", "of all transactions", FRAUD_CLR),
    ("Best AUC",            f"{best_result['auc']:.4f}",  best_name, ACCENT1),
    ("Best Avg Precision",  f"{best_result['ap']:.4f}",   best_name, ACCENT3),
    ("Features Used",       f"{len(FEATURES)}",   "engineered features",  ACCENT2),
    ("Models Evaluated",    f"{len(models)}",     "algorithms compared",  SAFE_CLR),
]
for i, (title, val, sub, col) in enumerate(kpis):
    x = 0.08 + i * 0.155
    rect = FancyBboxPatch((x-0.065, 0.05), 0.13, 0.88,
                           boxstyle="round,pad=0.02", linewidth=2,
                           edgecolor=col, facecolor=CARD_BG,
                           transform=kpi_ax.transAxes)
    kpi_ax.add_patch(rect)
    kpi_ax.text(x, 0.72, val, transform=kpi_ax.transAxes,
                ha='center', fontsize=22, fontweight='bold', color=col)
    kpi_ax.text(x, 0.50, title, transform=kpi_ax.transAxes,
                ha='center', fontsize=9, color=TEXT_LIGHT, fontweight='bold')
    kpi_ax.text(x, 0.27, sub, transform=kpi_ax.transAxes,
                ha='center', fontsize=8, color=TEXT_DIM, style='italic')

# ── Model Comparison Table ──
table_ax = fig6.add_axes([0.02, 0.07, 0.55, 0.55])
table_ax.axis('off')
table_ax.set_facecolor(DARK_BG)

cols = ['Model', 'AUC', 'Avg Precision', 'Fraud Precision', 'Fraud Recall', 'F1-Score']
rows = []
for name, res in results.items():
    r = res['report']['1']
    rows.append([name, f"{res['auc']:.4f}", f"{res['ap']:.4f}",
                 f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}"])

tbl = table_ax.table(cellText=rows, colLabels=cols, loc='center',
                      cellLoc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#1E2A3A')
    if row == 0:
        cell.set_facecolor(ACCENT4)
        cell.set_text_props(color=DARK_BG, fontweight='bold')
    else:
        model_name = rows[row-1][0]
        if model_name == best_name:
            cell.set_facecolor('#1A2A1A')
            cell.set_text_props(color=ACCENT1, fontweight='bold')
        else:
            cell.set_facecolor(CARD_BG)
            cell.set_text_props(color=TEXT_LIGHT)
table_ax.set_title("Model Comparison", color=ACCENT1, fontsize=13,
                   fontweight='bold', pad=10)

# ── Top features bar ──
top_ax = fig6.add_axes([0.60, 0.07, 0.38, 0.55])
top_ax.set_facecolor(CARD_BG)
top5_feats = feat_sorted[:8]
top5_imps  = imp_sorted[:8]
bar_c = [FRAUD_CLR, FRAUD_CLR, ACCENT2, ACCENT4, ACCENT4,
         ACCENT1, ACCENT1, ACCENT1][:len(top5_feats)]
h_bars = top_ax.barh(range(len(top5_feats))[::-1], top5_imps,
                     color=bar_c, height=0.65, edgecolor=DARK_BG)
for bar, imp in zip(h_bars, top5_imps):
    top_ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{imp:.3f}", va='center', fontsize=9, color=TEXT_LIGHT)
top_ax.set_yticks(range(len(top5_feats))[::-1])
top_ax.set_yticklabels(top5_feats, fontsize=9)
top_ax.set_title("Top 8 Fraud Signals", color=ACCENT1, fontsize=12,
                 fontweight='bold', pad=10)
top_ax.set_xlabel("Importance Score")
top_ax.grid(axis='x', alpha=0.3)
top_ax.spines[['top', 'right']].set_visible(False)

fig6.savefig('outputs/06_executive_summary.png',
             dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("✔  Figure 6 saved → 06_executive_summary.png")

fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
fig6.show()
print("\n" + "═"*60)
print("  ALL 6 VISUALIZATIONS SAVED SUCCESSFULLY")
print("═"*60)
