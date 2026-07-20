import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==================== CONFIG ====================
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.2f}".format)

# === UPDATE THIS PATH TO YOUR RAW FOLDER ===
DATA_DIR = r"D:\Office\Labmentrix\Project_9_Ford-Go-bike sharing\Ford-Go-bike sharing\New folder\Raw"

# ==================== 1. LOAD DATA ====================
print("🔍 Loading monthly files...")

# Flexible search in your Raw folder
files = sorted(glob.glob(os.path.join(DATA_DIR, "*fordgobike-tripdata.csv")))

print(f"Found {len(files)} files:")
for f in files:
    print("   ", os.path.basename(f))

if len(files) == 0:
    print("❌ No files found. Check the DATA_DIR path above.")
    exit()

dfs = [pd.read_csv(f, low_memory=False) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Combined dataset shape: {df.shape}")

# ==================== 2. INITIAL INSPECTION ====================
print("\n=== Dataset Info ===")
df.info()

print("\n=== Missing Values ===")
print(df.isna().sum().sort_values(ascending=False))

# ==================== 3. CLEANING & FEATURE ENGINEERING ====================
print("\n🧹 Cleaning & Engineering Features...")

df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

df = df.drop_duplicates()

# Plausible duration (1 min to 24 hrs)
df = df[(df['duration_sec'] >= 60) & (df['duration_sec'] <= 86400)].copy()

df['duration_min'] = df['duration_sec'] / 60.0
df['start_month'] = df['start_time'].dt.month
df['start_hour'] = df['start_time'].dt.hour
df['day_of_week'] = df['start_time'].dt.day_name()
df['is_weekend'] = df['start_time'].dt.weekday >= 5

df['age'] = 2018 - df['member_birth_year']
df.loc[(df['age'] < 10) | (df['age'] > 100), 'age'] = np.nan

df['member_gender'] = df['member_gender'].fillna('Unknown')
df['user_type'] = df['user_type'].fillna('Unknown')

print(f"Rows after cleaning: {len(df):,}")

# ==================== 4. SAVE CLEANED DATA ====================
df.to_csv(os.path.join(DATA_DIR, '..', 'ford_gobike_cleaned.csv'), index=False)
print("✅ Cleaned data saved as 'ford_gobike_cleaned.csv'")

# ==================== 5. VISUALIZATIONS ====================
# ==================== EXPANDED EDA: 12 KEY GRAPHS ====================
print("\n📊 Generating 12 comprehensive visualizations...")

output_dir = os.path.join(os.path.dirname(DATA_DIR), 'EDA_Charts')
os.makedirs(output_dir, exist_ok=True)

# 1. User Type 
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='user_type', hue='user_type', palette='Set2', legend=False)
plt.title('User Type Distribution')
plt.savefig(os.path.join(os.path.dirname(DATA_DIR), 'user_type.png'), dpi=200, bbox_inches='tight')
plt.show()

# 2. Gender Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='member_gender', hue='member_gender', palette='Set2', legend=False)
plt.title('Gender Distribution')
plt.savefig(os.path.join(output_dir, 'gender.png'), dpi=200, bbox_inches='tight')
plt.show()

# 3. Age Distribution
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age', bins=30, kde=True, color='skyblue')
plt.title('Rider Age Distribution')
plt.savefig(os.path.join(output_dir, 'age_dist.png'), dpi=200, bbox_inches='tight')
plt.show()

# 4. Monthly Usage Trend
plt.figure(figsize=(10,6))
monthly = df.groupby('start_month').size()
sns.barplot(x=monthly.index, y=monthly.values, palette='viridis')
plt.title('Monthly Trip Volume (2018)')
plt.xlabel('Month')
plt.ylabel('Number of Trips')
plt.savefig(os.path.join(output_dir, 'monthly_trend.png'), dpi=200, bbox_inches='tight')
plt.show()

# 5. Weekday vs Weekend
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='day_of_week', hue='is_weekend', palette='Set2')
plt.title('Trips by Day of Week')
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, 'weekday_pattern.png'), dpi=200, bbox_inches='tight')
plt.show()

# 6. Trip Duration Distribution
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='duration_min', bins=100, kde=True)
plt.xlim(0, 60)
plt.title('Trip Duration Distribution')
plt.savefig(os.path.join(output_dir, 'duration_hist.png'), dpi=200, bbox_inches='tight')
plt.show()

# 7. Duration by User Type 
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='user_type', y='duration_min', hue='user_type', palette='Set2', legend=False)
plt.ylim(0, 60)
plt.title('Trip Duration by User Type')
plt.savefig(os.path.join(os.path.dirname(DATA_DIR), 'duration_boxplot.png'), dpi=200, bbox_inches='tight')
plt.show()

# 8. Duration by Gender
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='member_gender', y='duration_min', hue='member_gender', palette='Set2', legend=False)
plt.ylim(0, 60)
plt.title('Trip Duration by Gender')
plt.savefig(os.path.join(output_dir, 'duration_gender.png'), dpi=200, bbox_inches='tight')
plt.show()

# 9. Hourly Heatmap (Multivariate)
plt.figure(figsize=(12,8))
heatmap_data = df.pivot_table(index='start_hour', columns='day_of_week', values='duration_sec', aggfunc='count')
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False)
plt.title('Trip Count Heatmap: Hour vs Day of Week')
plt.savefig(os.path.join(output_dir, 'hour_day_heatmap.png'), dpi=200, bbox_inches='tight')
plt.show()

# 10. Top 10 Start Stations
top_stations = df['start_station_name'].value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot(y=top_stations.index, x=top_stations.values, palette='viridis')
plt.title('Top 10 Start Stations')
plt.xlabel('Number of Trips')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_stations.png'), dpi=200, bbox_inches='tight')
plt.show()

# 11. Age vs Duration (Scatter - Sampled)
sample = df.dropna(subset=['age']).sample(n=5000, random_state=42)
plt.figure(figsize=(10,6))
sns.scatterplot(data=sample, x='age', y='duration_min', hue='user_type', alpha=0.6)
plt.ylim(0, 60)
plt.title('Age vs Trip Duration (Sampled)')
plt.savefig(os.path.join(output_dir, 'age_duration_scatter.png'), dpi=200, bbox_inches='tight')
plt.show()

# 12. User Type + Weekend Interaction
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='is_weekend', y='duration_min', hue='user_type', palette='Set2')
plt.title('Average Trip Duration: Weekend vs Weekday by User Type')
plt.savefig(os.path.join(output_dir, 'weekend_duration.png'), dpi=200, bbox_inches='tight')
plt.show()

print(f"\n🎯 All 12 charts saved in '{output_dir}' folder.")
print("EDA Complete with strong business coverage.")