import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'logs/throughput'

logfile_list = [f'{path}/infaas_v2.csv', f'{path}/clipper.csv', f'{path}/sommelier.csv',
                f'{path}/ilp.csv']

MARKERS_ON = True

hatches = ['-', '\\', '/', 'x', '+']
markers = ['.', 's', 'v', '^', 'x', '+', '*']
markersizes = [7, 3, 4, 4, 5, 6, 5]

algorithms = ['INFaaS-Accuracy', 'Clipper', 'Sommelier', 'Proteus']

colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C', '#8C564B',
          '#E377C2', 'tab:olive', 'tab:cyan']

fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [3, 1]})
fig.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.2)


color_idx = 0
clipper_accuracy = []
slo_violation_ratios = []
throughputs = []
effective_accuracies = []
accuracy_drops = []
y_cutoff = 0

infaas_y_intercept = 0
sommelier_y_intercept = 0
proteus_y_intercept = 0

for idx in range(len(logfile_list)):
    logfile = logfile_list[idx]
    
    algorithm = logfile.split('/')[-1].rstrip('.csv')

    df = pd.read_csv(logfile)

    original_df = copy.deepcopy(df)

    aggregated = df.groupby(df.index // 10).sum(numeric_only=True)
    aggregated = df.groupby(df.index // 10).mean(numeric_only=True)
    df = aggregated

    start_cutoff = 0

    time = df['simulation_time'].values[start_cutoff:]
    demand = df['demand'].values[start_cutoff:]
    throughput = df['throughput'].values[start_cutoff:]
    capacity = df['capacity'].values[start_cutoff:]

    y_cutoff = max(y_cutoff, max(demand))

    dropped = df['dropped'].values[start_cutoff:]
    late = df['late'].values[start_cutoff:]
    total_slo_violations = dropped + late

    successful = df['successful'].values[start_cutoff:]

    effective_accuracy = df['effective_accuracy'].values[start_cutoff:]
    total_accuracy = df['total_accuracy'].values[start_cutoff:]
    effective_accuracy = total_accuracy / successful

    if 'clipper' in algorithm:
        clipper_accuracy = effective_accuracy

    for i in range(len(successful)):
        if successful[i] > demand[i]:
            successful[i] = demand[i]

    difference = demand - successful - dropped

    slo_violation_ratio = (sum(original_df['demand']) - sum(original_df['successful']) + sum(original_df['late'])) / sum(original_df['demand'])
    slo_violation_ratios.append(slo_violation_ratio)

    throughputs.append(sum(original_df['successful']) / len(original_df['successful']))

    overall_effective_accuracy = sum(original_df['total_accuracy']) / sum(original_df['successful'])
    effective_accuracies.append(overall_effective_accuracy)

    max_accuracy_drop = 100 - min(effective_accuracy)
    accuracy_drops.append(max_accuracy_drop)

    time = time
    time = [x - time[0] for x in time]
    time = [x / time[-1] * 24 for x in time]

    if idx == 0:
        axs[0, 0].plot(time, demand, label='Demand', color=colors[color_idx],
                 marker=markers[color_idx])
        color_idx += 1

    if MARKERS_ON == True:
        axs[0, 0].plot(time, successful, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
        axs[1, 0].plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
        axs[2, 0].plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])

    if 'infaas' in algorithm:
        infaas_y_intercept = min(effective_accuracy)
    elif 'sommelier' in algorithm:
        sommelier_y_intercept = min(effective_accuracy)
    elif 'proteus' in algorithm:
        proteus_y_intercept = min(effective_accuracy)
    
    color_idx += 1

axs[0, 0].grid()
axs[1, 0].grid()
axs[2, 0].grid()

y_cutoff += 50


axs[0, 0].legend(loc='upper center', bbox_to_anchor=(0.65, 1.65), ncol=3, fontsize=14)

axs[0, 0].set_xticklabels([])
axs[0, 0].set_xticks(np.arange(0, 25, 4))
axs[0, 0].set_yticks(np.arange(0, y_cutoff + 50, 200))
axs[0, 0].set_ylabel('Throughput', fontsize=11)

axs[1, 0].set_xticks(np.arange(0, 25, 4))
axs[1, 0].set_xticklabels([])
axs[1, 0].set_yticks(np.arange(80, 104, 5))
axs[1, 0].set_ylabel('Effective Acc.', fontsize=11)

axs[2, 0].set_xticks(np.arange(0, 25, 4))
axs[2, 0].set_yticks(np.arange(0, 510, 100))
axs[2, 0].set_ylabel('SLO Violations', fontsize=11)

axs[2, 0].set_xlabel('Time (min)', fontsize=12)

axs[0, 1].set_ylabel('Avg. Throughput', fontsize=11)
axs[0, 1].bar(algorithms, throughputs, label=algorithms, color=colors[1:],
              hatch=hatches[1:], edgecolor='black')
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks(np.arange(0, 410, 100))

axs[1, 1].set_ylabel('Max. Acc. Drop', fontsize=11)
axs[1, 1].bar(algorithms, accuracy_drops, label=algorithms, color=colors[1:],
              hatch=hatches[1:], edgecolor='black')
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks(np.arange(0, 21, 5))

axs[2, 1].set_xlabel('Algorithm', fontsize=12)
axs[2, 1].set_ylabel('SLO Violation Ratio', fontsize=11)
axs[2, 1].bar(algorithms, slo_violation_ratios, label=algorithms, color=colors[1:],
              hatch=hatches[1:], edgecolor='black')
axs[2, 1].set_yticks(np.arange(0, 0.41, 0.1))
axs[2, 1].set_xticks([])

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

savefile = 'figures/timeseries_together.pdf'
plt.savefig(os.path.join(savefile), dpi=500, bbox_inches='tight')
print(f'Figure saved at {savefile}')
