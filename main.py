import os
import os.path as osp
import math
import time
import pandas as pd
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from model.pspnet import PSPNet, PSPLoss
from dataset.dataloader import make_datapath_list
from dataset.transforms import DataTransform, VOCDataset
from util.visuals import display_img
from util.metrics import PSPrate

import matplotlib.pyplot as plt

import wandb



wandb.login(key="e3fce9c3b09a787730aabf0c30a78c691f6dcb19", relogin=True)

ROOTPH = 'Updated_dataset'
TRAIN_TXT = '/ImageSets/Segmentation/train.txt'
VAL_TXT = '/ImageSets/Segmentation/val.txt'
EPOCH = 100
BATCH_TRAIN = 4
MINI_BACH_TRAIN = 2
BACH_VAL = 4
MINI_BACH_VAL = 2
STADY_IMAGE = 'stady_image'
WIGHTS_FILE = ''

# Logging all your results to wandb
run = wandb.init(
    # Set the project where this run will be logged
    entity="water-hyacinth-research-team",
    project="uncategorized",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 100,
    })



def train_model(net, dataloaders_dict, criterion, create_rate, scheduler, optimizer, num_epochs):
    train_loss_list = []
    val_loss_list = []
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    print(torch.cuda.is_available())

    # ネットワークをGPUへ
    net.to(device)
    #print(net)
    # summary(net,([8, 3, (475,475)]))

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # multiple minibatch
    batch_multiplier_train = MINI_BACH_TRAIN
    batch_multiplier_val = MINI_BACH_VAL

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:

            batch_size = dataloaders_dict[phase].batch_size

            if phase == 'train':
                net.train()  # モデルを訓練モードに
                optimizer.zero_grad()  # 最適化schedulerの更新
                optimizer.step()  # PyTorch 1.1.0以降では、逆の順序で呼び出す必要
                scheduler.step()

                print('（train）')

            else:
                net.eval()  # モデルを検証モードに
                print('-------------')
                print('（val）')

            count = 0  # multiple minibatch
            subset_count = 0
            # データローダーからminibatchずつ取り出すループ
            for imges, anno_class_imges in dataloaders_dict[phase]:

                imges_len = len(imges)
                img_stock = []
                for r in range(imges_len):

                    img = imges[r, :, :, :].numpy().transpose(2, 1, 0)
                    img = np.fliplr(img)
                    img = np.rot90(img, 1)

                    if r == imges_len - 1:
                        picup_img = img

                    img = img.transpose(2, 1, 0)
                    new_imge = torch.from_numpy(img.astype(np.float64)).clone()
                    img_stock.append(new_imge)

                new_imges = torch.stack(img_stock)

                # plt.imshow(new_imges[r, :, :, :].numpy().transpose(2, 1, 0))
                # plt.show()

                # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
                if new_imges.size()[0] == 1:
                    continue

                # GPUが使えるならGPUにデータを送る
                new_imges = new_imges.to(device, dtype=torch.float)
                anno_class_imges = anno_class_imges.to(device, dtype=torch.float)

                # multiple minibatchでのパラメータの更新
                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier_train

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = net(new_imges)
                    loss = criterion(outputs,
                                     anno_class_imges.long()) / batch_multiplier_train  # anno_class_imgesを１０進数に変換
                    #print(outputs)
                    tpft_array = create_rate.pixcel_count(outputs, anno_class_imges, phase)

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':

                        #accuracy loss 
                        accuracy = (tpft_array[0][0] + tpft_array[0][1]) / (tpft_array[0][0] + tpft_array[0][1]+ tpft_array[0][2] + tpft_array[0][3])
                        print(accuracy)
                        loss_accuracy = 1- accuracy
                        print("accuracy loss:", loss_accuracy* 0.3)
                        loss = loss + (loss_accuracy* 0.3)
                        
                        # 勾配の計算
                        loss.backward()

                        # multiple minibatch
                        count -= 1

                        # 学習対象の画像の表示
                        file_name = ROOTPH+'/'+STADY_IMAGE+'/epoch'+str(epoch)+'_'+phase+'_subset'+str(subset_count)+'.jpg'
                        display_img(file_name,epoch, create_rate, picup_img, 'train', subset_count)
                        subset_count += 1

                        # グラフ化
                        for tpft in tpft_array:
                            create_rate.pspresult(tpft[0], tpft[1], tpft[2], tpft[3], 'train')

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(iteration,
                                                                                               loss.item() / batch_size * batch_multiplier_train,
                                                                                               duration))
                            t_iter_start = time.time()
                            print(batch_multiplier_train)
                            epoch_train_loss += loss.item() * batch_multiplier_train
                            iteration += 1




                    # 検証時
                    else:

                        epoch_val_loss += loss.item() * batch_multiplier_val

                        # 学習対象の画像の表示
                        file_name = ROOTPH + '/' + STADY_IMAGE + '/epoch' + str(epoch) + '_' + phase + '_subset' + str(
                            subset_count) + '.jpg'

                        display_img(file_name,epoch, create_rate, picup_img, 'val', subset_count)
                        subset_count += 1

                        # グラフ化
                        for tpft in tpft_array:
                            create_rate.pspresult(tpft[0], tpft[1], tpft[2], tpft[3], 'val')

            # epochのphaseごとのlossと正解率
            t_epoch_finish = time.time()
            print('-------------')
            print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
                epoch + 1, epoch_train_loss / num_train_imgs, epoch_val_loss / num_val_imgs))
            print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
            t_epoch_start = time.time()

            # lossの可視化のリスト
            train_loss_list.append(epoch_train_loss / num_train_imgs)
            val_loss_list.append(epoch_val_loss / num_val_imgs)
            if ((epoch + 1) % 5 == 0):
                torch.save(net.state_dict(), ROOTPH + '/weights/pspnet50_' + str(epoch + 1) + '.pth')
                # ログを保存
            log_epoch = {'epoch': epoch + 1, 'train_loss': epoch_train_loss /
                                                           num_train_imgs, 'val_loss': epoch_val_loss / num_val_imgs}
            
            # weight_log
            # wandb.log(log_epoch)
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv("log_output.csv")

    # 最後のネットワークを保存する
    torch.save(net.state_dict(), ROOTPH + '/weights/' + WIGHTS_FILE + '/pspnet50_' + str(epoch + 1) + '.pth')
    return train_loss_list, val_loss_list


if __name__ == "__main__":


    try:
        # ファイルのパスリストを作成
        rootpath = ROOTPH
        train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
            rootpath=rootpath, train_text=TRAIN_TXT, val_text=VAL_TXT)

        # ROOTPH+'/'+STADY_IMAGE+'/epoch'+str(epoch)+'_'+phase+'_subset'+str(subset_count)+'.jpg'

        print(train_img_list[0])
        print(train_anno_list[0])
        print(val_img_list[0])
        print(val_anno_list[0])

        # PSPnetの用意
        net = PSPNet(n_classes=3)  # 150 or 2
        state_dict = torch.load(ROOTPH + "/weights/pspnet50_ADE20K.pth")
        #net.load_state_dict(state_dict)
        n_classes = 3
        net.decode_feature.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1,
                                                    padding=0)
        net.aux.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        # レイヤー確認

        # summary(net, input_size=(2,1,3))#input_size=(out_channels,H,W)

        """ データセット作成"""
        train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
            input_size=475))

        val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
            input_size=475))
        '''
        print('データの取り出し例')
        print("例１",train_dataset.__getitem__(0)[1].shape)
        print("例２",val_dataset.__getitem__(0)[1].shape)
        print("例３",val_dataset.__getitem__(0)[1])
        '''

        """データローダーの作成"""
        ## Creating Data Loader
        batch_size_train = BATCH_TRAIN  # ミニバッチのサイズを指定 (Making the mini batch size for training)
        batch_size_val = BACH_VAL       # Making the mini batch size for validation

        # Loading the training data
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=batch_size_train, shuffle=True)

        # Loading the validation data
        val_dataloader = data.DataLoader(
            val_dataset, batch_size=batch_size_val, shuffle=False)

        # 辞書オブジェクトにまとめる
        # Combine the train dataset and val dataset into dictionary object
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

        # 動作の確認

        batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
        imges, anno_class_imges = next(batch_iterator)  # 1番目の要素を取り出す
        #print(imges.size())  # torch.Size([8, 3, 475, 475])
        #print(anno_class_imges.size())  # torch.Size([8, 3, 475, 475])

        criterion = PSPLoss(aux_weight=0.4)
        create_rate = PSPrate()

        # スケジューラーを利用したepochごとの学習の変更
        # Change learning for each epoch using scheduler
        optimizer = optim.SGD([
            {'params': net.feature_conv.parameters(), 'lr': 1e-3},
            {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
            {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
            {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
            {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
            {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
            {'params': net.decode_feature.parameters(), 'lr': 1e-2},
            {'params': net.aux.parameters(), 'lr': 1e-2},
        ], momentum=0.9, weight_decay=0.0001)


        # スケジューラーの設定
        # Scheduler settings
        def lambda_epoch(epoch):
            max_epoch = 400
            return math.pow((1 - epoch / max_epoch), 0.9)


        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

        CUDA_LAUNCH_BLOCKING=1
        # 学習・検証を実行する
        # Training and Validation processe
        num_epochs = EPOCH
        train_loss_list, val_loss_list = train_model(net, dataloaders_dict, criterion, create_rate, scheduler, optimizer,
                                                    num_epochs)

        #plt.plot(train_loss_list, color='red')
        #plt.plot(val_loss_list, color='blue')
        #plt.show()
    except Exception as Error:
        print(Error)

