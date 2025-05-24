import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
df = pd.read_csv(dir + "data/training_set_VU_DM.csv")
output_dir = os.path.join(dir, "visualizations and tables")
os.makedirs(output_dir, exist_ok=True)

columns_to_vis = ['site_id', 'visitor_location_country_id', 
                  'visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_destination_id', 
                  'srch_length_of_stay', 'srch_booking_window','srch_adults_count', 'srch_children_count',
                  'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
                  'orig_destination_distance', 'random_bool', 'price_usd']

for column in columns_to_vis:
    print(column)
    print(df[column].describe())
    print(df[column].value_counts())
    
columns_histable = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_length_of_stay', 
                    'srch_booking_window','srch_adults_count', 'srch_children_count',
                  'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
                  'orig_destination_distance', 'random_bool', 'prop_brand_bool', 'prop_location_score1', 
                  'prop_location_score2', 'promotion_flag', 'prop_starrating', 'prop_review_score', 
                  'prop_log_historical_price', 'price_usd']

fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))  
fig.tight_layout(pad=4.0)  
axes = axes.flatten()

for i, column in enumerate(columns_histable):
    print(f"\nProcessing column: {column}")    
    sns.histplot(data=df, x=column, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {column}")    
    axes[i].tick_params(axis='x', rotation=45)

combined_filename = "combined_histograms.png"
save_path = os.path.join(output_dir, combined_filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\nSaved combined histograms: {save_path}")
print("\nAll histograms saved successfully!")

stats_df = pd.DataFrame()
for column in columns_histable:
    desc = df[column].describe().round(3).to_frame().T 
    desc['variable'] = column  
    stats_df = pd.concat([stats_df, desc], ignore_index=True)

stats_df = stats_df[['variable'] + [col for col in stats_df.columns if col != 'variable']]

stats_path = os.path.join(output_dir, "variable_statistics.csv")
stats_df.to_csv(stats_path, index=False)
print(f"Saved statistics table to: {stats_path}")

# correlation table with click_bool and booking_bool

targets = ['click_bool', 'booking_bool']
# Selects only numeric columns 
numeric_cols = df.select_dtypes(include=['number']).columns

if all(t in numeric_cols for t in targets):
    corr = df[numeric_cols].corr()[targets].sort_values(by=targets, ascending=False)
    corr.to_csv(os.path.join(output_dir, "target_correlations.csv"))
    print("Correlation analysis completed!")
else:
    print(f"Target columns {targets} not found in data")

plt.figure(figsize=(10, len(numeric_cols)//2))
sns.heatmap(corr, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1,
            linewidths=0.5)
plt.title("Correlation with Targets")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
plt.close()