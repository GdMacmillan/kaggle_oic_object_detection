gsutil -m rsync -r -d s3://open-images-dataset/train gs://3fc7db0a-d2aa-11e9-9747-0242ac1c0002/input/train
gsutil -m rsync -r -d s3://open-images-dataset/test gs://3fc7db0a-d2aa-11e9-9747-0242ac1c0002/input/test
gsutil -m rsync -r -d s3://open-images-dataset/validation gs://3fc7db0a-d2aa-11e9-9747-0242ac1c0002/input/valid
