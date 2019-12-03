import os
import gcsfs
import pandas as pd
import dask.dataframe as dd
import time
import json

file_name = os.path.basename(__file__).split('.')[0]

if __name__ == '__main__':
    with open('/data/config.json', 'r') as f:
        data = json.load(f)
        PROJECT = data['project']
        CREDENTIALS = os.environ[data['credentials']]
        read_bucket_name = data[file_name]['readBucket']
        input_file_name = data[file_name]['inputFile']

    FILE_PATH = os.path.join(read_bucket_name, input_file_name)

    start = time.time()

    # create filesystem object to handle our credentials
    fs = gcsfs.GCSFileSystem(project=PROJECT,
                             token=CREDENTIALS)

    # read with pandas
    with fs.open(FILE_PATH) as f:
        # reader = pd.read_csv(f, chunksize=100000)
        # df = pd.concat([chunk for chunk in reader])
        df = dd.read_csv(f, storage_options={'token': fs.session.credentials}, blocksize=25e6)
        print("Num rows: ", df.compute().shape[0])
    end = time.time()

    print("Wall time: ", (end - start))
