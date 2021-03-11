from plotting_utils import obtain_metrics
from plotting_utils import plot_acc_taskwise

path = "savedump"
prefix = "ResNet18_20_epochs"
all_metrics_multiple = dict()
for suffix in ["_imagenet", "_imagenet_ewc", "_imagenet_shuffled", "_imagenet_shuffled_ewc"]:
    all_metrics_multiple[suffix] = obtain_metrics(path, prefix = prefix, suffix = suffix)

for key in all_metrics_multiple.keys():
    print(f"key {key} len: {len(all_metrics_multiple[key])}")

plot_acc_taskwise(all_metrics_multiple, "ResNet18", save_to = "ResNet18_aggregated_plots.pdf")

# import matplotlib.pyplot as plt
# divs = 4
# fig = plt.figure()
# gs = fig.add_gridspec(ncols = divs, nrows = divs)
# axes = [fig.add_subplot(gs[div, div:]) for div in range(divs)]
# for row in range(divs):
#     axes[row].plot([1]*10*(divs - row), c = 'r')
#     axes[row].set_xlabel('', fontsize = 6)
# # axes[reconstr_task_no].tick_params(axis="y",direction="in", pad=-22)
# fig.set_figheight(10)
# fig.set_figwidth(10)
# # fig.tight_layout(pad = 0.3)
# plt.show()