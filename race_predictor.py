import fastf1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Setup cache
os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# 2026 races
races = [
    'Australia', 'China', 'Japan',
    'Miami', 'Canada', 'Monaco',
    'Spain', 'Austria'
]

print("Loading race data for ML training...")

all_data = []

for race in races:
    try:
        # Race results
        r_session = fastf1.get_session(2026, race, 'R')
        r_session.load(telemetry=False, weather=False, messages=False)

        # Qualifying results
        q_session = fastf1.get_session(2026, race, 'Q')
        q_session.load(telemetry=False, weather=False, messages=False)

        # Default weather values
        avg_temp = 30
        has_rain = 0

        results = r_session.results.copy()
        q_results = q_session.results[['Abbreviation', 'Position']].copy()
        q_results.columns = ['Abbreviation', 'QualPosition']

        merged = results.merge(q_results, on='Abbreviation', how='left')

        for _, row in merged.iterrows():
            try:
                finish_pos = pd.to_numeric(row['Position'], errors='coerce')
                grid_pos = pd.to_numeric(row['GridPosition'], errors='coerce')
                qual_pos = pd.to_numeric(row['QualPosition'], errors='coerce')

                if pd.isna(finish_pos) or pd.isna(grid_pos):
                    continue

                all_data.append({
                    'Race': race,
                    'Driver': row['Abbreviation'],
                    'GridPosition': grid_pos,
                    'QualPosition': qual_pos if not pd.isna(qual_pos) else grid_pos,
                    'FinishPosition': finish_pos,
                    'TrackTemp': avg_temp,
                    'HasRain': has_rain,
                    'PositionGain': grid_pos - finish_pos,
                })
            except:
                continue

        print(f"Loaded {race}")

    except Exception as e:
        print(f"Skipped {race}: {e}")

# Create dataframe
df = pd.DataFrame(all_data)
df = df.dropna()

print(f"\nTotal training samples: {len(df)}")

# ── Feature Engineering ──
le_driver = LabelEncoder()
le_race = LabelEncoder()
df['DriverEncoded'] = le_driver.fit_transform(df['Driver'])
df['RaceEncoded'] = le_race.fit_transform(df['Race'])

# Features and target
features = ['GridPosition', 'QualPosition', 'TrackTemp',
            'HasRain', 'DriverEncoded']
X = df[features]
y = df['FinishPosition']

# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Train Multiple Models ──
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

model_results = {}
print("\nTraining models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5,
                                 scoring='neg_mean_absolute_error')
    model_results[name] = {
        'model': model,
        'predictions': y_pred,
        'mae': mae,
        'r2': r2,
        'cv_mae': -cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    print(f"{name}: MAE={mae:.2f}, R²={r2:.3f}")

# Best model
best_model_name = min(model_results,
                       key=lambda x: model_results[x]['mae'])
best_model = model_results[best_model_name]['model']

# ── Feature Importance ──
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_)
    importances = importances / importances.sum()

# ── Predict Austrian GP ──
print("\nPredicting race outcomes...")
austrian_data = df[df['Race'] == 'Austria'].copy()
if len(austrian_data) > 0:
    pred_features = austrian_data[features].copy()
    predictions = best_model.predict(pred_features)
    austrian_data['PredictedPosition'] = predictions
    austrian_data['ActualPosition'] = austrian_data['FinishPosition']
    austrian_data = austrian_data.sort_values('PredictedPosition')

# ── Plotting ──
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('F1 2026 ML Race Outcome Predictor\nMachine Learning Analysis',
             fontsize=14, fontweight='bold')

model_names = list(model_results.keys())
maes = [model_results[m]['mae'] for m in model_names]
r2s = [model_results[m]['r2'] for m in model_names]
colors_m = ['#00D2BE', '#FF8700', '#DC0000']

# --- Plot 1: Model comparison ---
ax1 = axes[0, 0]
bars = ax1.bar(model_names, maes, color=colors_m,
               edgecolor='none', alpha=0.85)
ax1.set_ylabel('Mean Absolute Error (positions)', fontsize=10)
ax1.set_title('Model Accuracy Comparison\n(Lower MAE = More Accurate)',
              fontsize=11)
ax1.set_xticklabels(model_names, rotation=10, ha='right', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
best_idx = model_names.index(best_model_name)
bars[best_idx].set_edgecolor('yellow')
bars[best_idx].set_linewidth(2)
for bar, mae, r2 in zip(bars, maes, r2s):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.05,
             f'MAE: {mae:.2f}\nR²: {r2:.3f}',
             ha='center', fontsize=9, color='white')

# --- Plot 2: Predicted vs Actual ---
ax2 = axes[0, 1]
y_pred_best = model_results[best_model_name]['predictions']
ax2.scatter(y_test, y_pred_best, color='#00D2BE',
            alpha=0.6, s=60, zorder=5)
ax2.plot([1, 20], [1, 20], color='white', linestyle='--',
         linewidth=1.5, label='Perfect prediction')
ax2.fill_between([1, 20], [0, 19], [2, 21],
                  alpha=0.1, color='#00AA44', label='±1 position')
ax2.fill_between([1, 20], [-1, 18], [3, 22],
                  alpha=0.1, color='#FFF200', label='±2 positions')
ax2.set_xlabel('Actual Position', fontsize=10)
ax2.set_ylabel('Predicted Position', fontsize=10)
ax2.set_title(f'Predicted vs Actual\n{best_model_name}', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 21)
ax2.set_ylim(0, 21)

# --- Plot 3: Feature importance ---
ax3 = axes[0, 2]
feature_names = ['Grid Position', 'Qual Position',
                 'Track Temp', 'Rain', 'Driver']
colors_f = ['#FF3333', '#FF8700', '#FFF200', '#0067FF', '#00D2BE']
bars3 = ax3.barh(feature_names, importances * 100,
                  color=colors_f, edgecolor='none', height=0.6)
ax3.set_xlabel('Feature Importance (%)', fontsize=10)
ax3.set_title(f'Feature Importance\n({best_model_name})', fontsize=11)
ax3.grid(True, alpha=0.3, axis='x')
ax3.invert_yaxis()
for bar, val in zip(bars3, importances * 100):
    ax3.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9, color='white')

# --- Plot 4: Austrian GP prediction vs actual ---
ax4 = axes[1, 0]
if len(austrian_data) > 0:
    top_drivers = austrian_data.head(10)
    x_pos = np.arange(len(top_drivers))
    width = 0.35
    ax4.bar(x_pos - width/2, top_drivers['ActualPosition'],
            width, label='Actual', color='#FF8700',
            edgecolor='none', alpha=0.85)
    ax4.bar(x_pos + width/2, top_drivers['PredictedPosition'],
            width, label='Predicted', color='#00D2BE',
            edgecolor='none', alpha=0.85)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(top_drivers['Driver'],
                         rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Race Position', fontsize=10)
    ax4.set_title('Austrian GP — Predicted vs Actual\n(Top 10)',
                  fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.invert_yaxis()

# --- Plot 5: Cross validation ---
ax5 = axes[1, 1]
cv_maes = [model_results[m]['cv_mae'] for m in model_names]
cv_stds = [model_results[m]['cv_std'] for m in model_names]
bars5 = ax5.bar(model_names, cv_maes, color=colors_m,
                edgecolor='none', alpha=0.85,
                yerr=cv_stds, capsize=5,
                error_kw={'color': 'white', 'linewidth': 2})
ax5.set_ylabel('Cross-Validation MAE', fontsize=10)
ax5.set_title('5-Fold Cross Validation\n(Lower = Better)', fontsize=11)
ax5.set_xticklabels(model_names, rotation=10, ha='right', fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
for bar, mae, std in zip(bars5, cv_maes, cv_stds):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + std + 0.05,
             f'{mae:.2f}±{std:.2f}',
             ha='center', fontsize=9, color='white')

# --- Plot 6: Grid vs Finish ---
ax6 = axes[1, 2]
scatter = ax6.scatter(df['GridPosition'], df['FinishPosition'],
                       c=df['PositionGain'],
                       cmap='RdYlGn', s=40, alpha=0.6,
                       vmin=-10, vmax=10, zorder=5)
plt.colorbar(scatter, ax=ax6,
             label='Positions Gained (Green = Gained)')
ax6.plot([1, 20], [1, 20], color='white',
         linestyle='--', linewidth=1.5,
         label='Grid = Finish', alpha=0.7)
corr = df['GridPosition'].corr(df['FinishPosition'])
ax6.set_xlabel('Grid Position', fontsize=10)
ax6.set_ylabel('Finish Position', fontsize=10)
ax6.set_title(f'Grid vs Finish Position\nCorrelation: {corr:.3f}',
              fontsize=11)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('race_predictor.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("F1 2026 RACE OUTCOME PREDICTOR — FINAL REPORT")
print("="*50)
print(f"Training samples: {len(df)}")
print(f"Best Model: {best_model_name}")
for name in model_names:
    r = model_results[name]
    print(f"  {name}: MAE={r['mae']:.2f}, R²={r['r2']:.3f}")
