#!/usr/bin/env python
!pip install imgviz
!pip install labelme
import argparse
import glob
import os
import os.path as osp

import imgviz
import numpy as np

import labelme
import shutil

FOLDER_NAME = "C:/Users/TADELE/Documents/PSPSNet_semantic segmentation/Annotation_project"
FILE_NAME = FOLDER_NAME + "/" + "Image"
NEWDATASET_NAME = FOLDER_NAME + "/"+"Dataset"


IMAGESET_PATH = NEWDATASET_NAME + '/ImageSets'
RESULTIMAGE_PATH = NEWDATASET_NAME + '/result_image'
RESULTJSON_PATH = NEWDATASET_NAME + '/result_json'
STADYIMAGE_PATH = NEWDATASET_NAME + '/stady_image'
WEIGHT_PATH = NEWDATASET_NAME + '/weights'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #parser.add_argument("input_dir", help="input annotated directory", default="C:\\Users\\Test\\Documents\\data")
    #parser.add_argument("output_dir", help="output dataset directory", default="data_dataset_voc")
    parser.add_argument("--labels", help="labels file", default="labels.txt")
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )

    args = parser.parse_args(args=[])

    input_dir = FILE_NAME
    output_dir = NEWDATASET_NAME

    if osp.exists(output_dir):
        print("Output directory already exists:", output_dir)
        return
    
    os.makedirs(output_dir)
    os.makedirs(osp.join(output_dir, "JPEGImages"))
    os.makedirs(osp.join(output_dir, "SegmentationClass"))
    os.makedirs(osp.join(output_dir, "SegmentationClassPNG"))

    if not args.noviz:
        os.makedirs(
            osp.join(output_dir, "SegmentationClassVisualization")
        )
    
    os.makedirs(osp.join(output_dir, "SegmentationObject"))
    os.makedirs(osp.join(output_dir, "SegmentationObjectPNG"))

    if not args.noviz:
        os.makedirs(
            osp.join(output_dir, "SegmentationObjectVisualization")
        )
    
    print("Creating dataset:", output_dir)

    class_names = []
    class_name_to_id = {}
    print(args.labels)
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        print(class_name)
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "Background"
        class_names.append(class_name)
    
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(output_dir, "class_names.txt")

    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)
    print('hello')
    for filename in glob.glob(osp.join(input_dir, "*.json")):
        print("Generating dataset from:", filename)
            
        label_file = labelme.LabelFile(filename=filename)
        
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(output_dir, "JPEGImages", base + ".jpg")
        out_cls_file = osp.join(
            output_dir, "SegmentationClass", base + ".npy"
        )
        out_clsp_file = osp.join(
            output_dir, "SegmentationClassPNG", base + ".png"
        )
        if not args.noviz:
            out_clsv_file = osp.join(
                output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )
        out_ins_file = osp.join(
            output_dir, "SegmentationObject", base + ".npy"
        )
        out_insp_file = osp.join(
            output_dir, "SegmentationObjectPNG", base + ".png"
        )
        if not args.noviz:
            out_insv_file = osp.join(
                output_dir,
                "SegmentationObjectVisualization",
                base + ".jpg",
            )

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        ins[cls == -1] = 0  # ignore it.

        # class labels
        labelme.utils.lblsave(out_clsp_file, cls)
        np.save(out_cls_file, cls)
        if not args.noviz:
            clsv = imgviz.label2rgb(
                cls,
                imgviz.rgb2gray(img),
                label_names=class_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_clsv_file, clsv)

        # instance label
        labelme.utils.lblsave(out_insp_file, ins)
        np.save(out_ins_file, ins)
        if not args.noviz:
            instance_ids = np.unique(ins)
            instance_names = [str(i) for i in range(max(instance_ids) + 1)]
            insv = imgviz.label2rgb(
                ins,
                imgviz.rgb2gray(img),
                label_names=instance_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(out_insv_file, insv)
    
    return


if __name__ == "__main__":
    main()

if os.path.exists(IMAGESET_PATH) == False:
        os.mkdir(IMAGESET_PATH)
if os.path.exists(RESULTIMAGE_PATH) == False:
        os.mkdir(RESULTIMAGE_PATH)
if os.path.exists(RESULTJSON_PATH) == False:
        os.mkdir(RESULTJSON_PATH)
if os.path.exists(STADYIMAGE_PATH) == False:
        os.mkdir(STADYIMAGE_PATH)
if os.path.exists(WEIGHT_PATH) == False:
        os.mkdir(WEIGHT_PATH)
        shutil.copy('pspnet50_ADE20K.pth',WEIGHT_PATH)
