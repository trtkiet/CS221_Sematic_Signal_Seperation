import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

folder_path = './results'  # Adjust if files are in a different directory
# 1. Define the files to load
files = ['intfloat__e5-large-v2.jsonl', 'all-MiniLM-L6-v2.jsonl']

# 2. Load and parse the data
data = []
for file in files:
    full_path = os.path.join(folder_path, file)
    if not os.path.exists(full_path):
        print(f"Warning: {full_path} not found. Skipping.")
        continue
        
    with open(file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Flatten the nested 'results' dictionary
                if 'results' in item:
                    for k, v in item['results'].items():
                        item[k] = v
                    del item['results']
                item['source_file'] = file
                data.append(item)
            except json.JSONDecodeError:
                continue

if not data:
    print("No data found. Please check your files.")
else:
    df = pd.DataFrame(data)

    # 3. Clean up source file names for cleaner legend labels
    def clean_source(x):
        if 'intfloat' in x:
            return 'e5-large-v2'
        if 'MiniLM' in x:
            return 'MiniLM-L6-v2'
        return x

    df['Embedding'] = df['source_file'].apply(clean_source)

    # 4. Create Visualizations
    sns.set_theme(style="whitegrid")

    metrics = ['diversity', 'wec_ex', 'wec_in']
    metric_titles = ['Diversity', 'Coherence (External)', 'Coherence (Internal)']

    # Create a separate set of plots for each dataset found in the files
    unique_datasets = df['dataset'].unique()

    for dataset in unique_datasets:
        df_sub = df[df['dataset'] == dataset]
        
        # Setup a 1x3 grid of plots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Results for Dataset: {dataset}', fontsize=20)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric in df_sub.columns:
                # Lineplot automatically aggregates over multiple seeds (showing mean + confidence interval)
                sns.lineplot(
                    data=df_sub, 
                    x='n_topics', 
                    y=metric, 
                    hue='model',        # Color by Model (S3, S3_angular, etc.)
                    style='Embedding',  # Dash/Marker by Source File (Embedding)
                    markers=True, 
                    dashes=False, 
                    ax=ax,
                    linewidth=2
                )
                ax.set_title(metric_titles[i], fontsize=15)
                ax.set_xlabel('Number of Topics', fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                # Improve legend placement
                ax.legend(title='Model / Embedding', fontsize=10)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        output_filename = f'{dataset.replace(" ", "_")}_results.png'
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Saved plot: {output_filename}")
        plt.show()