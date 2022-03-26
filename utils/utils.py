import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img,cmap="gray")
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Pred_mask (c_{i + 1})')
            ax[i + 1].imshow(mask[i,:, :],cmap="gray")
    else:
        ax[1].set_title(f'Predict mask')
        ax[1].imshow(mask,cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.show()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count