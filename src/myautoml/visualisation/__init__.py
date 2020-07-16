import numpy as np


def plot_grouped_bar(ax, groups_of_bars, x_labels, group_labels, colors):
    """Plot a grouped bar chart"""
    n_bars_in_group = len(groups_of_bars[0])
    bar_width = 1 / (n_bars_in_group + 1)

    x_pos = []
    for group_num, group in enumerate(groups_of_bars):
        x_pos = [x + group_num * bar_width for x in np.arange(len(group))]
        bars = ax.bar(x_pos, group, color=colors[group_num], width=bar_width, label=group_labels[group_num])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02, round(height, 2), ha='center', va='bottom')

    ax.set_xticks(x_pos, x_labels)
    ax.legend()

    return ax
