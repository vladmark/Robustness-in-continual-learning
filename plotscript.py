from plotting_utils import obtain_metrics
from plotting_utils import plot_acc_taskwise
from plotting_utils import plot_acc_average_tasks
from plotting_utils import plot_train_rout_conf_entr

path = "savedump"
task = "_cl_t5_c8"
model_type = "VGG11"
all_metrics_multiple = dict()
suffixes = ["", "_ewc", "_shuffled"]
for suffix in ["_imagenet" + task + elt for elt in suffixes]:
    all_metrics_multiple[suffix] = obtain_metrics(path, prefix = model_type + "_20_epochs", suffix = suffix, no_tasks = 5)



    # plot_train_rout_conf_entr(all_metrics_multiple[suffix], model_type + suffix, save_to = model_type + task + suffix + "_classwise_analysis.pdf")

plot_acc_average_tasks(all_metrics_multiple, model_type + task, save_to = model_type + task + "_avg_accuracies_all_tasks.pdf")
# plot_acc_taskwise(all_metrics_multiple, "ResNet18"+task, save_to = model_type + task + "_aggregated_plots_with_specdec.pdf")

# for key in all_metrics_multiple.keys():
    # print(f"key {key} {all_metrics_multiple[key][0]['test_confusion']}")

# print(all_metrics_multiple["_imagenet"+task][4].keys())



