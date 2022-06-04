import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(standardized, filtered, masks, mixed):
    labels = ['shufflenetv2_x0_5', 'resnet50', 'efficientnet_b5', 'efficientnet_b3', 'convnext_tiny', 'convnext_small', 'convnext_base']

    standardized_l = list(map(lambda x: round(x * 100,1) , [standardized['shufflenetv2_x0_5'][0], standardized['resnet50'][0], standardized['efficientnet_b5'][0], standardized['efficientnet_b3'][0], standardized['convnext_tiny'][0], standardized['convnext_small'][0], standardized['convnext_base'][0]]))
    filtered_l = list(map(lambda x: round(x * 100,1) , [filtered['shufflenetv2_x0_5'][0], filtered['resnet50'][0], filtered['efficientnet_b5'][0], filtered['efficientnet_b3'][0], filtered['convnext_tiny'][0], filtered['convnext_small'][0], filtered['convnext_base'][0]]))
    masks_l = list(map(lambda x: round(x * 100,1) , [masks['shufflenetv2_x0_5'][0], masks['resnet50'][0], masks['efficientnet_b5'][0], masks['efficientnet_b3'][0], masks['convnext_tiny'][0], masks['convnext_small'][0], masks['convnext_base'][0]]))
    mixed_l = list(map(lambda x: round(x * 100,1) , [mixed['shufflenetv2_x0_5'][0], mixed['resnet50'][0], mixed['efficientnet_b5'][0], mixed['efficientnet_b3'][0], mixed['convnext_tiny'][0], mixed['convnext_small'][0], mixed['convnext_base'][0]]))

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
