import os
import os.path as osp



"""ファイルのパスリストの作成"""
def make_datapath_list(rootpath,train_text,val_text ):
    '''
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    '''

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template_train = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    imgpath_template_val = osp.join(rootpath, 'JPEGImages', '%s.jpg')

    annopath_template_train = osp.join(rootpath, 'SegmentationClassPNG', '%s.png')
    annopath_template_val = osp.join(rootpath, 'SegmentationClassPNG', '%s.png')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + train_text)
    val_id_names = osp.join(rootpath + val_text)

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template_train % file_id)  # 画像のパス
        anno_path = (annopath_template_train % file_id)  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template_val % file_id)  # 画像のパス
        anno_path = (annopath_template_val % file_id)  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

#前処理