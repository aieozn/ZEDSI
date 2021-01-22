import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def draw_grouped_bar_chart(categories, categories_false_positives, categories_true_negatives):
    labels = categories

    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
        
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig.set_figwidth(15)
    rects1 = ax1.bar(x - width/2, categories_false_positives, width, label='False positive')
    rects2 = ax1.bar(x + width/2, categories_true_negatives, width, label='True negative')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Invalid classification')
    ax1.set_title('Wyniki dla poszczególnych typów')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()


    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    sum_fp = sum(categories_false_positives)
    sum_tn = sum(categories_true_negatives)
    ax2.bar(["Suma false positive", "Suma false negative"], [sum_fp, sum_tn])
    ax2.set_title("Ostateczny wynik dla kategorii")
    ax2.set_ylabel("Hamming loss")

    plt.show()

def draw_progress_chart(loss, val_loss, accuracy, val_accuracy, suptitle = None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(20)
    fig.set_figheight(6)
    ax1.plot(loss)
    ax1.plot(val_loss)
    ax1.legend(["Train", "Validation"])
    ax1.set_title('Loss')
    ax1.set_ylim([0, 1])

    ax2.plot(accuracy)
    ax2.plot(val_accuracy)
    ax2.legend(["Train", "Validation"])
    ax2.set_title('Accuracy')
    ax2.set_ylim([0, 1])

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    plt.show()

def draw_types_result(categories, categories_results, hamming_loss):
    plt.figure(figsize=(20, 8))

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.bar(categories, categories_results)
    ax1.set_title("Wyniki dla poszczególnych typów")
    ax1.set_ylabel("% poprawnych wyników")
    ax1.set_xlabel("Category")
    ax1.set_ylim([0, 1])

    ax2.bar(["All"], hamming_loss)
    ax2.set_title("Ostateczny wynik")
    ax2.set_ylabel("Hamming loss")
    ax2.set_ylim([0, 1])

    plt.show()