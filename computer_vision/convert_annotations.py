
from PIL import Image
# from pymongo import MongoClient
from collections import defaultdict
from tempfile import NamedTemporaryFile
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
import gcsfs
import os
import io
import json
import tarfile
import shutil
import time

file_name = os.path.basename(__file__).split('.')[0]

pool_size = multiprocessing.cpu_count()
print("Using pool of %d cpus" % pool_size)

fields = ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']

def get_im_size(fs, remote_fp):
    """
    input -
        fs filesystem object
        fp filepath or path object
    output - width, height
    """
    with fs.open(remote_fp, 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        return img.size

def write_main(fs, remote_fp, data):
    """
    input -
        fs filesystem object
        fp filepath or path object
        data object
    output - bool
    ref: https://stackoverflow.com/questions/39109180/dumping-json-directly-into-a-tarfile
    """
    tmp_file = NamedTemporaryFile()
    filename = tmp_file.name
    with io.BytesIO() as out_stream, tarfile.open(filename, 'w|gz') as tar_file:
        out_stream.write(json.dumps(data).encode())
        out_stream.seek(0)
        info = tarfile.TarInfo("data")
        info.size = len(out_stream.getbuffer())
        tar_file.addfile(info, out_stream)

    fs.put(filename, remote_fp)

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

    with open('/data/config.json', 'r') as f: # uncomment for container runtime
    # with open('computer_vision/config.json', 'r') as f:
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
    output_fp = os.path.join(read_bucket_name, output_file_name)
    # create filesystem object to handle our credentials
    fs = gcsfs.GCSFileSystem(project=project,
                             token=credentials)

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

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start the load operations and mark each future with its filepath
        future_to_fp = {executor.submit(get_im_size, fs, fp): fp for fp in fs.glob(str(imgs_path/'*.jpg'))}
        for future in concurrent.futures.as_completed(future_to_fp):
            fp = future_to_fp[future]
            try:
                width, height = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (fp, exc))
            else:
                filepath = Path(fp)
                image = {
                    "file_name": filepath.name,
                    "height": height,
                    "width": width,
                    "id": filepath.stem,
                }
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

    with fs.open(labels_fp) as f:
        labels = pd.read_csv(f, names=["Class"], index_col=0)
    labels['supercategory'] = labels.Class
    labels['name'] = labels.Class
    labels['id'] = labels.index

    output_dict['categories'].extend(labels.to_dict('records'))
    print('initiating main upload')
    write_main(fs, output_fp, output_dict)
    print('output file upload complete')
    # send(out.name)
    end = time.time()

    print("Wall time: ", (end - start))

if __name__ == '__main__':
    main()
