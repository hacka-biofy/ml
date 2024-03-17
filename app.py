# https://www.mdpi.com/2306-5354/9/7/271

import os
import shutil

from flask import Flask, request

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO


BACTERIA_NAMES = ['ecoli', 'saureus', 'paureginosa', 'paeruginosa']

IMAGES_PATH = 'Petri_plates'
OUTPUT_PATH = 'seg_petri_plates'

IMG_SHAPE = (300, 300, 3)


app = Flask(__name__)

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)

def _image_segmentation():
    ...

def download_image_from_storage(image_url):
    response = requests.get(image_url, stream=True)
    img = Image.open(response.raw)
    return img



def image_feature_engineering(image_buffer):
  resized_img = image_buffer

  resized_img = resized_img.thumbnail(IMG_SHAPE[:2])
  vec_img = image.img_to_array(resized_img)

  features = vgg_model.predict(image_buffer)
  feature_matrix = features.reshape(features.shape[0], -1)

  feature_matrix

  traindata[0][0].shape


  feature_matrix[0]


  n_features = len(feature_matrix[0])
  n_features


  client = QdrantClient('localhost', port=6333)


  client.create_collection(
      collection_name='test_collection',
      vectors_config=VectorParams(size=n_features, distance=Distance.DOT),
  )

  class_indices_ = traindata.class_indices
  labels_ = traindata.labels

  indices_class_ = {v: k for k, v in class_indices_.items()}

  indices_class_


  points = [
      PointStruct(
          id=i,
          vector=vec.tolist(),
          payload={
              'label': indices_class_.get(labels_[i]),
              'img_uri': traindata.filepaths[i],
          },
      )
      for i, vec in tqdm(enumerate(feature_matrix[:100]))
  ]

  

def split_images_dataset(full_images_list, output_path=OUTPUT_PATH):
    for bn in BACTERIA_NAMES:

        _family = [
            image_name for image_name in full_images_list if bn in image_name
        ]

        new_filepath = os.path.join(output_path, bn)

        os.makedirs(new_filepath, exist_ok=True)

        new_family_paths = [
            os.path.join(output_path, bn, image_name.split('/')[-1])
            for image_name in _family
        ]

        for i, orig_file in enumerate(_family):
            shutil.copy(orig_file, new_family_paths[i])


@app.route("/image_segmentation", methods=['GET', 'POST'])
def image_segmentation():
    payload = request.json
    oracle_s3_uri = payload.get('image_uri')
    
    image_ = download_image_from_storage(oracle_s3_uri)

    image_feature_engineering(image_)

    return "<p>Hello, World!</p>"


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port='3002')