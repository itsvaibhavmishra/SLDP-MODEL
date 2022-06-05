SCRIPTS_PATH = 'Tensorflow/scripts' # generated tf records
APIMODEL_PATH = 'Tensorflow/models' # tf object detection model from github
WORKSPACE_PATH = 'Tensorflow/workspace'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
MODEL_PATH = WORKSPACE_PATH + '/models' # will store trained models
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/' # stores checkpoints from trained models

labels = [
    {'name':'Hello', 'id':1},
    {'name':'Friend', 'id':2},
    {'name':'Yes', 'id':3},
    {'name':'No', 'id':4},
    {'name':'Thank You', 'id':5},
    {'name':'Ok', 'id':6},
    {'name':'Bathroom', 'id':7},
    {'name':'Please', 'id':8},
    {'name':'1000 Years of Death', 'id':9},
]

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' # name of custom model

# Importing dependencies for updating config

import tensorflow as tf
from object_detection.utils import config_util # for path configuration
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Loading trained model from checkpoint
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vz_utils # for making bounding box
from object_detection.builders import model_builder # builds model from checkpoint

# Real-Time Detections using OpenCV
# importing dependencies
import cv2
import numpy as np

CONFIG_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/pipeline.config'
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH) # getting config into notebook for making changes

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig() # Template pipeline config
with tf.io.gfile.GFile(CONFIG_PATH, 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 9 # number of different types of model need to be configured i.e. 9
pipeline_config.train_config.batch_size = 4 # data processed dependent on hardware capacity
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = 'detection' # Changing from classification to detection
pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record'] # tf records
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record'] # tf records

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial() # ckpt-21 is the latest checkpoint

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image) # resizing image to 320x320
    prediction_dict = detection_model.predict(image, shapes) # making predictions
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

# Setting up capture
cap = cv2.VideoCapture(0) # '0' is the device number for my camera
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print('''\n
    \n
    ================================ BOOT-UP ================================= 
    \n''')

while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    vz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

    cv2.imshow('SLDP-Training-Model',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('''\n================================ EXITING ================================= \n''')
        cap.release()
        break