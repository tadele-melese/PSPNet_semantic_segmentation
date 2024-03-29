from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import torch

from dataset.dataloader import make_datapath_list 
from dataset.transforms import DataTransform


# ファイルパスリスト作成
ROOTPH = 'Updated_dataset'
TRAIN_TXT = '/ImageSets/Segmentation/train.txt'
VAL_TXT = '/ImageSets/Segmentation/val.txt'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath=ROOTPH, train_text=TRAIN_TXT, val_text=VAL_TXT)

# 後ほどアノテーション画像のみを使用する

from model.pspnet import PSPNet

net = PSPNet(n_classes=3)

# 学習済みパラメータをロード
state_dict = torch.load("./Updated_dataset/weights/pspnet50_95.pth",
                        map_location={'cuda:0': 'cpu'})
net.load_state_dict(state_dict)

print('ネットワーク設定完了：学習済みの重みをロードしました')

# 1. 元画像の表示
image_file_path = "./Updated_dataset/JPEGImages/DJI_0269.jpg"
img = Image.open(image_file_path)   # [高さ][幅][色RGB]
img_width, img_height = img.size
plt.imshow(img)
plt.show()

# 2. 前処理クラスの作成
#color_mean = (0.485, 0.456, 0.406)
#color_std = (0.229, 0.224, 0.225)
transform = DataTransform(input_size=475)

# 3. 前処理
# 適当なアノテーション画像を用意し、さらにカラーパレットの情報を抜き出す
anno_file_path = val_anno_list[0]
anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
p_palette = anno_class_img.getpalette()
phase = "val"
img, anno_class_img = transform(phase, img, anno_class_img)


# 4. PSPNetで推論する
net.eval()
x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])

outputs = net(x.float())
y = outputs[0]  # AuxLoss側は無視 yのサイズはtorch.Size([1, 21, 475, 475])


# 5. PSPNetの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
y = y[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
y = np.argmax(y, axis=0)
#plt.imshow(y)
#plt.show()
anno_class_img = Image.fromarray(np.uint8(y), mode="P")
anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
anno_class_img.putpalette(p_palette)
plt.imshow(anno_class_img)
plt.show()


# 6. 画像を透過させて重ねる
trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
anno_class_img = anno_class_img.convert('RGBA')  # カラーパレット形式をRGBAに変換

for x in range(img_width):
    for y in range(img_height):
        # 推論結果画像のピクセルデータを取得
        pixel = anno_class_img.getpixel((x, y))
        r, g, b, a = pixel

        # (0, 0, 0)の背景ならそのままにして透過させる
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            continue
        else:
            # それ以外の色は用意した画像にピクセルを書き込む
            trans_img.putpixel((x, y), (r, g, b, 150))
            # 150は透過度の大きさを指定している

img = Image.open(image_file_path)   # [高さ][幅][色RGB]
result = Image.alpha_composite(img.convert('RGBA'), trans_img)
plt.imshow(result)
plt.show()
