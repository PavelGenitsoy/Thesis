import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
fig = plt.figure(figsize=(8, 4.8))
ax = fig.add_subplot()
x_best, x, y_best, y = [], [], [], []
column_name = []
prev, it = 0, 0
for acc, time in zip(eval_sorted_by_time.values[0], eval_sorted_by_time.values[1]):
    if acc > prev:
        prev = acc
        column_name.append(eval_sorted_by_time.columns.to_list()[it])
        y_best.append(round(acc * 100, 2))
        x_best.append(math.log(time, 33))
    else:
        y.append(round(acc * 100, 2))
        x.append(math.log(time, 33))
    it += 1
ax.step(x_best, y_best, '-b', where='post')
plt.plot(x_best, y_best, '--', color='grey', alpha=0.7)
plt.scatter(x_best, y_best, c='green', s=100, label='best models')
it = 0
for name, point in zip(column_name, zip(x_best, y_best)):
    x_cor, y_cor = point
    if not it:
        x_cor -= 0.04
    elif it == 3:
        x_cor -= 0.09
    else:
        x_cor -= 0.07
    if it not in (0, 4):
        y_cor += 1
    else:
        y_cor -= 2
    ax.annotate(name, (x_cor, y_cor))
    it += 1
plt.scatter(x, y, c='red', s=20, label='other models')
labels = [item.get_text() for item in ax.get_xticklabels()]
for i, tick in enumerate([0.2, 0.4, 0.6, 0.8, 1, 1.2], start=1):
    labels[i] = round(np.power(33, tick), 2)
ax.set_xticklabels(labels)
ax.grid()
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Time (sec)')
ax.set_title('Accuracy plot for inference time on test (2100 images)')
ax.legend(loc='lower right')
plt.ylim(50, 90)

plt.show()
# plt.savefig('data/accuracy_plot_for_best_inference_time.png', bbox_inches='tight')
"""


def main(standardized, filtered, masks, mixed):
    labels = ['shufflenetv2_x0_5', 'resnet50', 'efficientnet_b5', 'efficientnet_b3', 'convnext_tiny', 'convnext_small', 'convnext_base']

    standardized_l = list(map(lambda x: round(x * 100, 1), [standardized[i][0] for i in labels]))
    filtered_l = list(map(lambda x: round(x * 100, 1), [filtered[i][0] for i in labels]))
    masks_l = list(map(lambda x: round(x * 100, 1), [masks[i][0] for i in labels]))
    mixed_l = list(map(lambda x: round(x * 100, 1), [mixed[i][0] for i in labels]))

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 4.8))

    rects1 = ax.bar(x - 3 * width/2, standardized_l, width, label='standardized', color='#005aff')
    rects2 = ax.bar(x - width/2, filtered_l, width, label='filtered', color='white', edgecolor='#005aff', hatch='///')
    rects3 = ax.bar(x + width/2, masks_l, width, label='masks', color='#f1c232')
    rects4 = ax.bar(x + 3 * width/2, mixed_l, width, label='mixed', color='white', edgecolor='#f1c232', hatch='---')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Models')
    ax.set_title('Accuracy by dataset and model')
    ax.set_xticks(x, labels)

    ax.legend(loc='upper left', frameon=False)

    ax.bar_label(rects1, padding=2)
    ax.bar_label(rects2, padding=2)
    ax.bar_label(rects3, padding=2)
    ax.bar_label(rects4, padding=2)

    fig.tight_layout()
    plt.ylim(35, 95)
    # plt.show()
    plt.savefig('data/top5+2models.png', bbox_inches='tight')


if __name__ == '__main__':
    standardized = pd.read_csv('data/results/history_standardized_dataset.csv')
    filtered = pd.read_csv('data/results/history_filtered_with_cnn_standardized_dataset.csv')
    masks = pd.read_csv('data/results/history_masks_dataset.csv')
    mixed = pd.read_csv('data/results/history_mixed_dataset.csv')

    main(standardized, filtered, masks, mixed)
