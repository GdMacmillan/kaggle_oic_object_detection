
from PIL import Image
# from pymongo import MongoClient
from collections import defaultdict
from tempfile import NamedTemporaryFile
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import numpy as np
import multiprocessing
import gcsfs
import os
import io
import json
import time

file_name = os.path.basename(__file__).split('.')[0]

pool_size = multiprocessing.cpu_count()
print("num cpus: ", pool_size)
manager = multiprocessing.Manager()

fields = ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = get_image_metadata(func, args)
        output.put(result)

def get_image_metadata(func, args):
    width, height = func(*args)
    filepath = Path(args[1])
    image = {
        "file_name": filepath.name,
        "height": height,
        "width": width,
        "id": filepath.stem,
    }
    return image
    # return '%s is processing image: %s' % \
    #     (multiprocessing.current_process().name, image)

def get_im_size(fs, fp):
    """
    input - filepath or path object
    output - width, height
    """
    with fs.open(fp, 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        return img.size

def calc_bounding_box(row, im_width, im_height):
    """
    input - row with bounding box corner x min/max and y min/max
            image width in num pixels, height in num pixels
    output x, y, width and height of bounding box in num pixels
            area of bounding box in pixels
    """
    x, y = int(row['XMin'] * im_width), int(row['YMin'] * im_height)
    width = int((row['XMax'] - row['XMin']) * im_width)
    height = int((row['YMax'] - row['YMin']) * im_width)
    area = height * width

    return pd.Series({'bbox':[x, y, width, height], 'area': area})

def main():
    start = time.time()

    output_dict = defaultdict(list)
    with NamedTemporaryFile('w') as out:
        # with open('/data/config.json', 'r') as f: # uncomment for container runtime
        with open('computer_vision/config.json', 'r') as f:
            data = json.load(f) # uncomment for container runtime
            project = data['project']
            credentials = os.environ[data['credentials']]
            read_bucket_name = data[file_name]['readBucket']
            input_file_name = data[file_name]['inputFile']
            output_file_name = data[file_name]['outputFile']
            labels_file_name = data[file_name]['labelsFile']

        print('read config complete')
        imgs_path = Path(read_bucket_name)
        annotations_fp = os.path.join(read_bucket_name, input_file_name)
        labels_fp = os.path.join(read_bucket_name, labels_file_name)
        # create filesystem object to handle our credentials
        fs = gcsfs.GCSFileSystem(project=project,
                                 token=credentials)

        img_fps = [(get_im_size, (fs, fp)) for fp in fs.glob(str(imgs_path/'*.jpg'))]
        # create queues
        task_queue = multiprocessing.Queue()
        done_queue = multiprocessing.Queue()
        # submit tasks
        for task in img_fps: # TASKS1
            task_queue.put(task)

        for i in range(pool_size):
            multiprocessing.Process(target=worker, args=(task_queue, done_queue)).start()
        print('starting worker processes complete')
        # # read annotations into dask distributed dataframe
        # with fs.open(annotations_fp) as f:
        #     ddf = dd.read_csv(f, storage_options={'token': fs.session.credentials}, blocksize=25e6)
        #     print('read annotations complete')
        # # precompute group annotations by image id
        # groups = ddf.groupby('ImageID').compute().groups # groups of indexes
        # print('number of annotation groups found: ', len(groups.keys()))

        # read annotations
        with fs.open(annotations_fp) as f:
            reader = pd.read_csv(f, chunksize=100000)
            df = pd.concat([chunk for chunk in reader])
            print('read annotations complete')
        # precompute group annotations by image id
        groups = df.groupby('ImageID').groups # groups of indexes
        print('number of annotation groups found: ', len(groups.keys()))

        # get results  and convert annotations
        print('unordered results:')
        for i in range(len(img_fps)):
            # print('\t', done_queue.get())
            image = done_queue.get()

            width, height = image['width'], image['height']
            # anno = ddf.loc[groups[filepath.stem]].compute() # dask distributed
            anno = df.loc[groups[filepath.stem]]
            anno = anno.merge(anno.apply(calc_bounding_box, args=(width, height), axis=1), on=anno.index, validate='one_to_one')
            anno['iscrowd'] = anno.IsGroupOf
            anno['category_id'] = anno.LabelName
            anno['image_id'] = anno.ImageID
            anno['id'] = anno.key_0
            anno['segmentation'] = np.empty((len(anno), 0)).tolist()
            annotations = anno[fields].to_dict('records')

            output_dict['images'].append(image)
            output_dict['annotations'].extend(annotations)

        # Tell child processes to stop
        for i in range(pool_size):
            task_queue.put('STOP')

        with fs.open(labels_fp) as f:
            labels = pd.read_csv(f)
        labels['supercategory'] = labels.Class
        labels['name'] = labels.Class
        labels['id'] = labels.index

        output_dict['categories'].extend(labels.to_dict('records'))

        json.dump(output_dict, out)
        out.flush()

    # send(out.name)
    end = time.time()

    print("Wall time: ", (end - start))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
