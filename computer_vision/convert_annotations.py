
from PIL import Image
from pymongo import MongoClient
from collections import defaultdict
from pathlib import Path
import shutil
import multiprocessing
import threading
import os
import json
import time


POOL_SIZE = multiprocessing.cpu_count()
PROJECT = ""

annotation_fields = ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']


DB_NAME = ""
COLLECTION_NAME = ""

client = MongoClient()
db = client[DB_NAME]
coll = db[COLLECTION_NAME]


def get_im_size(fp):
    """
    input - filepath or path object
    output - width, height
    """
    with Image.open(fp, 'r') as img:
        return img.size

def calc_bounding_box(row, im_width, im_height):
    """
    input - row with bounding box corner x min/max and y min/max
            image width in num pixels, height in num pixels
    output - x, y, width and height of bounding box in num pixels
            area of bounding box in pixels
    """
    x, y = int(row['XMin'] * im_width), int(row['YMin'] * im_height)
    width = int((row['XMax'] - row['XMin']) * im_width)
    height = int((row['YMax'] - row['YMin']) * im_width)
    area = height * width

    return pd.Series({'bbox':[x, y, width, height], 'area': area})


def class AnnotationConverter(object):

    def get_img_fps_iter(self):
        return list((self.IMG_dir_path).glob('*.jpg'))

    def convert_annotations(self, img_fps):
        opt = defaultdict(list)

        for fp in img_fps:
            width, height = get_im_size(fp)
            image = {
                "file_name": fp.name,
                "height": height,
                "width": width,
                "id": fp.stem,
            }
            opt['images'].append(image)

            # convert annotations
            anno = self.df.loc[groups[fp.stem]]
            anno = anno.merge(anno.apply(calc_bounding_box, args=(width, height), axis=1), on=anno.index, validate='one_to_one')
            anno['iscrowd'] = anno.IsGroupOf
            anno['category_id'] = anno.LabelName
            anno['image_id'] = anno.ImageID
            anno['id'] = anno.key_0
            anno['segmentation'] = np.empty((len(anno), 0)).tolist()

            opt['annotations'].extend(
                anno[annotation_fields].to_dict('records')
            )

        labels['supercategory'] = labels.Class
        labels['name'] = labels.Class
        labels['id'] = labels.index

        opt['categories'].extend(labels.to_dict('records'))

        return opt

    def main(self):
        # TODO: check that generator is compatible with multiprocessing change iterator if necessary
        pool = multiprocessing.Pool(processes=POOL_SIZE)
        rslt = pool.map(func=self.convert_annotations, self.get_img_fps_iter())
        for d in rslt:
            for k, v in d.items():
                z[k].extend(v)


if __name__ == '__main__':
    converter = AnnotationConverter()
    converter.main()
