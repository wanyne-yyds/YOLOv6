#coding:utf-8
 
 
import os
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path 

path2 = "/data_ssd/ckn_ssd/Data_trainset/YOLOv6-CocoAndYOLOFormat-HOD-220914/" # 输出文件夹
classes = ["safety belt", "person", "deputy person", "both hands wheel", "single hands wheel", "not hands wheel", "dark", "bright", "dark phone", "bright phone", "ignore"]
train_xml_dir = "/data_ssd/ckn_ssd/Data_trainset/YOLO_Vehicles/YOLOXdatasetsHOD/Annotations/HOD-Muilt-Class/train/" # train xml文件
val_xml_dir = "/data_ssd/ckn_ssd/Data_trainset/YOLO_Vehicles/YOLOXdatasetsHOD/Annotations/HOD-Muilt-Class/val/" # train xml文件
train_img_dir = "/data_ssd/ckn_ssd/Data_trainset/YOLO_Vehicles/YOLOXdatasetsHOD/JPEGImages/HOD-Muilt-Class/train" # 图片
val_img_dir = "/data_ssd/ckn_ssd/Data_trainset/YOLO_Vehicles/YOLOXdatasetsHOD/JPEGImages/HOD-Muilt-Class/val" # 图片

# train_ratio = 1.0 # 训练集的比例

START_BOUNDING_BOX_ID = 1

def get(root, name):
    return root.findall(name)

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars
  
def convert(xml_list, json_file, yolox_format_path):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    for index, line in enumerate(xml_list):
        # print("Processing %s"%(line))
        xml_f = line
        txt_f = os.path.join(yolox_format_path, os.path.basename(xml_f)[:-3] + 'txt')
        txt_content = open(txt_f, "w")
        tree = ET.parse(xml_f)
        root = tree.getroot()
        filename = os.path.basename(xml_f)[:-4] + ".jpg"
        imgpath = str(xml_f).replace('Annotations', 'JPEGImages')[:-4] + ".jpg"
        if not os.path.isfile(imgpath):
            filename = os.path.basename(imgpath)[:-4] + ".png"
            imgpath = imgpath[:-4] + ".png"
        
        imgoutpath = yolox_format_path.replace('labels', 'images')
        shutil.copy(imgpath, imgoutpath)

        image_id = 20190000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            if category not in categories:
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories) + 1
                print("[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(category, pre_define_categories, new_id))
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert(xmax > xmin), "xmax <= xmin, {}".format(line)
            assert(ymax > ymin), "ymax <= ymin, {}".format(line)

            dw_yoloformat = 1./ width
            dh_yoloformat = 1./ height
            x_yoloformat = (xmin + xmax) / 2.0
            y_yoloformat = (ymin + ymax) / 2.0
            w_yoloformat = xmax - xmin
            h_yoloformat = ymax - ymin
            x_yoloformat = x_yoloformat * dw_yoloformat
            w_yoloformat = w_yoloformat * dw_yoloformat
            y_yoloformat = y_yoloformat * dh_yoloformat
            h_yoloformat = h_yoloformat * dh_yoloformat
            yolo_content = "%d %f %f %f %f\n" %(category_id, x_yoloformat, y_yoloformat, w_yoloformat, h_yoloformat)
            txt_content.write(yolo_content)

            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        txt_content.close()

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories), all_categories.keys(), len(pre_define_categories), pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())

if __name__ == '__main__':

    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
    # pre_define_categories = {'a1': 1, 'a3': 2, 'a6': 3, 'a9': 4, "a10": 5}
    only_care_pre_define_categories = True
    # only_care_pre_define_categories = False

    if os.path.exists(path2 + "/annotations"):
        shutil.rmtree(path2 + "/annotations")
    os.makedirs(path2 + "/annotations")
    if os.path.exists(path2 + "/images"):
        shutil.rmtree(path2 + "/images/train2017")
        shutil.rmtree(path2 + "/images/val2017")
    os.makedirs(path2 + "/images/train2017", exist_ok=True)
    os.makedirs(path2 + "/images/val2017", exist_ok=True)
    if os.path.exists(path2 + "/labels"):
        shutil.rmtree(path2 +"/labels/train2017")
        shutil.rmtree(path2 +"/labels/val2017")
    os.makedirs(path2 + "/labels/train2017", exist_ok=True)
    os.makedirs(path2 + "/labels/val2017", exist_ok=True)

    save_json_train = path2 + 'annotations/instances_train2017.json'
    save_json_val = path2 + 'annotations/instances_val2017.json'

    train_xml_list = list(str(i) for i in Path(train_xml_dir).rglob("**/*.xml"))
    train_xml_list = np.sort(train_xml_list)
    np.random.seed(100)
    np.random.shuffle(train_xml_list)

    val_xml_list = list(str(i) for i in Path(val_xml_dir).rglob("**/*.xml"))
    val_xml_list = np.sort(val_xml_list)
    np.random.seed(100)
    np.random.shuffle(val_xml_list)

    convert(train_xml_list, save_json_train, path2 + "/labels/train2017")
    convert(val_xml_list, save_json_val, path2 + "/labels/val2017")

