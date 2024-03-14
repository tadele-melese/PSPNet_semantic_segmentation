import torch
import numpy as np

from PIL import Image
import wandb
from matplotlib import pyplot as plt
from pathlib import Path

class PSPrate():

    def __init__(self):
        self.train_result_img = 0
        self.val_result_img = 0

    def pixcel_count(self, outputs, target, phase):

        device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        output = outputs[0]  # AuxLoss側は無視　yサイズはtorch.Size()
        output = output.to("cpu")  # CPU／GPUの切り替え
        target = target.to("cpu")  # CPU／GPUの切り替え
        TPFN_array = []

        with torch.no_grad():
            for i in range(len(output)):

                true_positive = 0
                true_negative = 0
                false_positive = 0
                false_negative = 0

                out_img = output[i, :, :, :].numpy()
                out_img = np.argmax(out_img, axis=0)  # 一番大きい要素のインデックスを返す(ピクセル毎に確信度が最大のクラスを求める。ここで８枚の画像からの学習結果が1枚でる。
                tar_img = target[i, :, :].numpy()
                #print("Predicted Image")
                #plt.imshow(out_img)
                #plt.show()
                #print("-----------------")
                #print("Ground truth Image")
                #plt.imshow(tar_img)
                #plt.show()
                #print("The shape of predict vs groundtruth", out_img.shape, tar_img.shape)

                # for x in range(475):
                #     for y in range(475):
                #         if out_img[x, y] == 1 and tar_img[x, y] == 1:
                #             true_positive += 1
                #         elif out_img[x, y] == 1 and tar_img[x, y] == 0:
                #             false_positive += 1
                #         elif out_img[x, y] == 0 and tar_img[x, y] == 1:
                #             false_negative += 1
                #         elif out_img[x, y] == 0 and tar_img[x, y] == 0:
                #             true_negative += 1

                # TPFN_array.append([true_positive, true_negative, false_positive, false_negative])

        train_class_img = Image.fromarray(255 - np.uint8(out_img) * 255, mode="P")  # (out_img[i]*255).astype(np.uint8)
        train_class_img.putpalette([255, 255, 0])  # 黄色に設定
        train_class_img = train_class_img.convert('RGB')
        print("Training Image")
        plt.imshow(train_class_img)
        plt.show()
        anno_class_img = Image.fromarray(255 - np.uint8(tar_img) * 255,
                                         mode="P")  # (tar_img[j]*255).astype(np.uint8)#黒白反転
        anno_class_img = anno_class_img.resize((475, 475), Image.NEAREST)
        anno_class_img.putpalette([0, 255, 255])  # 黄色に設定
        anno_class_img = anno_class_img.convert('RGB')
        print("Annotation Image")
        plt.imshow(anno_class_img)
        plt.show()

        result_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))

        for x in range(475):
            for y in range(475):
                # 学習結果画像とアノテーションデータ画像のピクセルデータを取得
                # Getting the training data and annotation data image's pixel 
                ##黄色(学習結果) Yellow is the training results
                pixel_show = train_class_img.getpixel((x, y))
                #print(pixel_show)
                ##青色(作成したアノテーションデータ) Blue is the ground truth 
                pixel = anno_class_img.getpixel((x, y)) # Getting pixel from annotation image
                #print(pixel)

                if pixel == (0, 255, 255):  # 作成したアノテーションデータ ground truth pixel data
                    if pixel_show == (255, 255, 0):  # 学習結果画像 training pixel data
                        result_img.putpixel((x, y), (255, 255, 0, 150))  # yellow
                        true_positive += 1
                        continue
                        # True Positive
                    else:
                        result_img.putpixel((x, y), (0, 255, 255, 150))  # aqua
                        false_positive += 1
                        continue
                        # False Positive
                else:
                    if pixel_show == (255, 255, 0):  # 学習結果画像
                        result_img.putpixel((x, y), (255, 0, 0, 150))  # red
                        false_negative += 1
                        continue
                       # True Negative
                    else:
                         result_img.putpixel((x, y), (0, 0, 0, 50))  # black
                         true_negative += 1
                        # black(white)
                        #False Negative
                        #continue

            if phase == 'train':
                self.train_result_img = result_img
            
            else:
                self.val_result_img = result_img

        TPFN_array.append([true_positive, true_negative, false_positive, false_negative])
        plt.imshow(self.train_result_img)
        plt.show()
        return TPFN_array

    def pspresult(self, tp, tn, fp, fn, phase):

        precision = 0
        recall = 0

        if (tp + tn + fp + fn) != 0:

            #wandb.log({'正解率・正確さ('+phase+')':(tp+tn)/(tp+tn+fp+fn)*100})
            wandb.log({'Accuracy (' + phase + ')': (tp + tn) / (tp + tn + fp + fn) * 100})
            #pass
        else:
            wandb.log({'Accuracy ('+phase+')':0})
            #pass
        '''
        全体のデータの中で正しく分類できたTP とTNがどれだけあるかという指標。
        '''

        if (tp + fp) != 0:

            precision = tp / (tp + fp) * 100
            # wandb.log({'精度・適合率('+phase+')':tp/(tp+fp)*100})
            wandb.log({'Precision(' + phase + ')': precision})
            #pass
        else:
            # wandb.log({'精度・適合率('+phase+')':0})
            wandb.log({'Precision (' + phase + ')': precision})
            pass
        '''
        Positive と分類されたデータ(TP + FP)の中で実際にPositiveだったデータ(TP)数の割合。
        '''

        if (tp + fn) != 0:
            recall = tp / (tp + fn) * 100
            # wandb.log({'再現率・真陽性率('+phase+')':tp/(tp+fn)*100})
            wandb.log({'Recall (' + phase + ')': recall})
            pass
        else:
            # wandb.log({'再現率・真陽性率('+phase+')':0})
            wandb.log({'Recall (' + phase + ')': recall})
            pass
        '''
        取りこぼし無くPositive なデータを正しくPositiveと推測できているかどうか。
        '''

        if (fp + tn) != 0:
            # wandb.log({'真陰性率('+phase+')':tn/(fp+tn)*100})
            wandb.log({'true negative rate(' + phase + ')': tn / (fp + tn) * 100})
            pass
        else:
            # wandb.log({'真陰性率('+phase+')':0})
            wandb.log({' true negative rate  (' + phase + ')': 0})
            pass
        '''
        取りこぼし無くNegative なデータを正しくNegativeと推測できているかどうか。
        '''

        if (tp + fn) != 0:
            # wandb.log({'偽陰性率('+phase+')':fn/(tp+fn)*100})
            wandb.log({'false negative rate('+phase+')':fn/(tp+fn)*100})
            pass
        else:
            # wandb.log({'偽陰性率('+phase+')':0})
            wandb.log({'false negative rate('+phase+')':0})
            pass
        '''
        実際にはPositive であるサンプルの内、Negativeであると判定されたクラスの割合。
        '''

        if (fp + tn) != 0:
            # wandb.log({'偽陽性率('+phase+')':fp/(fp+tn)*100})
            wandb.log({' false positive rate ('+phase+')':fp/(fp+tn)*100})
            pass
        else:
            # wandb.log({'偽陽性率('+phase+')':0})
            wandb.log({'false positive rate('+phase+')':0})
            pass
        '''
        実際にはNegative であるサンプルの内、Positiveであると判定されたクラスの割合。
        '''

        F1 = 2 * (precision * recall) / (precision + recall)

        wandb.log({'F1 Score ('+phase+')':F1})

        return True




def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec




def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)





#@threaded
def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


#@threaded
def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
