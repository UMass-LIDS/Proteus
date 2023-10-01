import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'logs/throughput'

logfile_list = [f'{path}/infaas_v2.csv', f'{path}/clipper.csv', f'{path}/sommelier.csv',
                f'{path}/ilp.csv']

MARKERS_ON = True

markers = ['.', 's', 'v', '^', 'x', '+', '*']
markersizes = [7, 3, 4, 4, 5, 6, 5]
algorithms = ['INFaaS-Accuracy', 'Clipper', 'Sommelier', 'Proteus']
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C', '#8C564B',
          '#E377C2', 'tab:olive', 'tab:cyan']

fig, (ax1, ax2, ax3) = plt.subplots(3)
color_idx = 0
clipper_accuracy = []
slo_violation_ratios = []
y_cutoff = 0
for idx in range(len(logfile_list)):
    logfile = logfile_list[idx]
    
    algorithm = logfile.split('/')[-1].rstrip('.csv')

    df = pd.read_csv(logfile)

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

    slo_violation_ratio = (sum(df['demand']) - sum(df['successful'])) / sum(df['demand'])
    slo_violation_ratios.append(slo_violation_ratio)

    time = time
    time = [x - time[0] for x in time]
    time = [x / time[-1] * 24 for x in time]

    if idx == 0:
        ax1.plot(time, demand, label='Demand', color=colors[color_idx],
                 marker=markers[color_idx], markersize=markersizes[color_idx])
        color_idx += 1
    if MARKERS_ON == True:
        ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
        ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
        ax3.plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
    else:
        ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx])
        ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx])
        ax3.plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx])

        print(f'algorithm: {algorithm}, slo_violation_ratio: {slo_violation_ratio}')
    color_idx += 1
ax1.grid()
ax2.grid()
ax3.grid()

y_cutoff += 200

ax1.legend(loc='upper center', bbox_to_anchor=(0.45, 1.75), ncol=3, fontsize=12)

ax1.set_xticklabels([])
ax1.set_xticks(np.arange(0, 25, 4))
ax1.set_yticks(np.arange(0, y_cutoff + 50, 300))
ax1.set_ylabel('Throughput', fontsize=11)

ax2.set_xticks(np.arange(0, 25, 4))
ax2.set_xticklabels([])
ax2.set_yticks(np.arange(80, 104, 5))
ax2.set_ylabel('Effective Acc.', fontsize=11)

ax3.set_xticks(np.arange(0, 25, 4))
ax3.set_yticks(np.arange(0, 510, 100))
ax3.set_ylabel('SLO Violations', fontsize=11)

ax3.set_xlabel('Time (min)', fontsize=12)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

savefile = 'figures/bursty.pdf'
plt.savefig(os.path.join(savefile), dpi=500, bbox_inches='tight')
print(f'Figure saved at {savefile}')
