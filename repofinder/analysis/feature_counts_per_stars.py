import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

def plot_feature_distribution_by_star_bucket(
    df, features, star_col='stargazers_count', ax=None,
    title_prefix=None, acronym=None, hide_ylabel=False, ylim=None,
    order=None
):
    feature_display_names = {
        'description': 'Description',
        'readme': 'README',
        'license': 'License',
        'code_of_conduct_file': 'Code of Conduct',
        'contributing': 'Contributing Guide',
        'security_policy': 'Security Policy',
        'issue_templates': 'Issue Templates',
        'pull_request_template': 'PR Template'
    }
    # Reverse mapping: display name -> feature key
    display_to_key = {v: k for k, v in feature_display_names.items()}

    def get_star_bucket(stars):
        if stars <= 100:
            return '0–100'
        elif stars <= 500:
            return '101–500'
        else:
            return '>500'

    df = df.copy()
    df['star_bucket'] = df[star_col].apply(get_star_bucket)
    total_repositories = len(df)
    star_buckets = ['0–100', '101–500', '>500']

    # If order is given, map display names back to keys
    if order is not None:
        try:
            feature_keys = [display_to_key[name] for name in order]
        except KeyError as e:
            raise ValueError(f"Display name {e} in order not found in display names dictionary")
    else:
        feature_keys = features

    results = []
    for feature in feature_keys:
        for bucket in star_buckets:
            subset = df[df['star_bucket'] == bucket]
            total = len(subset)
            count = subset[feature].notna().sum()
            pct = (count / total) * 100 if total > 0 else 0
            results.append({
                'Feature': feature,
                'Star Bucket': bucket,
                'Percent': pct,
                'Count': count
            })

    plot_df = pd.DataFrame(results)
    x = np.arange(len(feature_keys))
    width = 0.25
    cmap = cm.get_cmap('tab20')
    colors = [cmap(i) for i in range(3)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    for i, bucket in enumerate(star_buckets):
        bucket_data = plot_df[plot_df['Star Bucket'] == bucket]
        bar_positions = x + (i - 1) * width
        bars = ax.bar(
            bar_positions,
            bucket_data['Percent'],
            width,
            label=bucket,
            color=colors[i]
        )

        # Add raw count labels on top
        for bar, count in zip(bars, bucket_data['Count']):
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{count}',
                            (bar.get_x() + bar.get_width() / 2, height + 2),
                            ha='center', va='bottom',
                            fontsize=8, color='black')

    # Use order display names if provided, else map keys to display names
    if order is not None:
        display_labels = order
    else:
        display_labels = [feature_display_names.get(f, f) for f in feature_keys]

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    if not hide_ylabel:
        ax.set_ylabel("% of Repos with Feature", fontsize=25)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)

    ax.set_xlabel("Feature", fontsize=25)
    ax.set_ylim(0, ylim if ylim else 108)
    ax.set_xlim(-0.5, len(feature_keys) - 0.3)

    if title_prefix and acronym:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=28,
            loc="left",
            pad=20
        )

    ax.legend(title='Stars', fontsize=13, title_fontsize=14)
