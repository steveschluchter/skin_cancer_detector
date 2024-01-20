import pandas as pd
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, save_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import sys


BENIGN_TYPES = ['nevus', 'seborrheic keratosis', 'pigmented benign keratosis', 'solar lentigo', 'dermatofribroma',
                        'vascular lesion', 'lentigo nos', 'lichenoid keratosis', 'lentigo simplex', 'aimp', 'angioma',
                        'neurofibroma', 'scar', 'verucca', 'acrochordon', 'angiofibroma', 'fibrous papule',
                        'cafe-au-lait macule', 'angiokeratoma', 'clear cell acanthoma']

MALIGNANT_TYPES = ['melanoma', 'basal cell carcinoma', 'actinic keratosis', 'squamous cell carcinoma',
                           'melanoma metastasis', 'atypical melanocytic proliferation', 'atypical spitz tumor']


bcn_20000_df = pd.read_csv('metadata/bcn20000_metadata_2023-08-28.csv')
challenge_2016_test_df = pd.read_csv('metadata/challenge-2016-test_metadata_2023-08-28.csv')
challenge_2016_train_df = pd.read_csv('metadata/challenge-2016-training_metadata_2023-08-28.csv')
challenge_2017_test_df = pd.read_csv('metadata/challenge-2017-test_metadata_2023-08-28.csv')
challenge_2017_training_df = pd.read_csv('metadata/challenge-2017-training_metadata_2023-08-28.csv')
challenge_2017_validation_df = pd.read_csv('metadata/challenge-2017-validation_metadata_2023-08-28.csv')
challenge_2018_task_1_2_test_df = pd.read_csv('metadata/challenge-2018-task-1-2-test_metadata_2023-08-29.csv')
challenge_2018_task_1_2_validation_df = pd.read_csv('metadata/challenge-2018-task-1-2-validation_metadata_2023-08-29.csv')
challenge_2018_task_1_2_training_df = pd.read_csv('metadata/challenge-2018-task-1-2-training_metadata_2023-08-29.csv')
challenge_2018_task_3_training_df = pd.read_csv('metadata/challenge-2018-task-3-training_metadata_2023-08-29.csv')
challenge_2018_task_3_validation_df = pd.read_csv('metadata/challenge-2018-task-3-validation_metadata_2023-08-29.csv')
challenge_2019_training_df = pd.read_csv('metadata/challenge-2019-training_metadata_2023-08-28.csv')
challenge_2020_training_metadata_df = pd.read_csv('metadata/challenge-2020-training_metadata_2023-08-28.csv')
ham10000_metadata_2023_df = pd.read_csv('metadata/ham10000_metadata_2023-08-30.csv')
prove_ai_metadata_2023_df = pd.read_csv('metadata/prove-ai_metadata_2023-08-30.csv')


challenge_2016_test_df = challenge_2016_test_df[['isic_id','benign_malignant','diagnosis']]
challenge_2016_train_df = challenge_2016_train_df[['isic_id','benign_malignant','diagnosis']]
challenge_2017_training_df = challenge_2017_training_df[['isic_id','benign_malignant','diagnosis']]
challenge_2017_validation_df = challenge_2017_validation_df[['isic_id','benign_malignant','diagnosis']]
challenge_2017_test_df = challenge_2017_test_df[['isic_id','benign_malignant','diagnosis']]
challenge_2018_task_1_2_test_df = challenge_2018_task_1_2_test_df[['isic_id','benign_malignant','diagnosis']]
challenge_2018_task_1_2_training_df = challenge_2018_task_1_2_training_df[['isic_id','benign_malignant','diagnosis']]
challenge_2018_task_1_2_validation_df = challenge_2018_task_1_2_validation_df[['isic_id','benign_malignant','diagnosis']]
challenge_2018_task_3_validation_df = challenge_2018_task_3_validation_df[['isic_id','benign_malignant','diagnosis']]
challenge_2019_training_df = challenge_2019_training_df[['isic_id','benign_malignant','diagnosis']]
challenge_2020_training_metadata_df = challenge_2020_training_metadata_df[['isic_id','benign_malignant','diagnosis']]
ham10000_metadata_2023_df = ham10000_metadata_2023_df[['isic_id','benign_malignant','diagnosis']]
prove_ai_metadata_2023_df = prove_ai_metadata_2023_df[['isic_id','benign_malignant','diagnosis']]
bcn_20000_df = bcn_20000_df[['isic_id', 'benign_malignant','diagnosis']]


changed_file_name_prefixes = ['BCN_2020','HAM_10000','2016_test','2016_train','2017_test','2017_train', '2017_validation', '2018_task_1-2_test',
                              '2018_task_1-2-validation','2018_task_3_training', '2019_data', '2020_data_training','PROVe-AI']

bcn_20000_df = bcn_20000_df[(bcn_20000_df['diagnosis'] == 'seborrheic keratosis') | (bcn_20000_df['diagnosis'] == 'pigmented benign keratosis') |
                            (bcn_20000_df['diagnosis'] == 'nevus') | (bcn_20000_df['diagnosis'] == 'solar lentigo') | (bcn_20000_df['diagnosis'] == 'dermatofribroma') |
                            (bcn_20000_df['diagnosis'] == 'vascular lesion') | (bcn_20000_df['diagnosis'] == 'lentigo nos') | (bcn_20000_df['diagnosis'] == 'lichenoid keratosis') |
                            (bcn_20000_df['diagnosis'] == 'lentigo simplex') | (bcn_20000_df['diagnosis'] == 'aimp') | (bcn_20000_df['diagnosis'] == 'angioma') |
                            (bcn_20000_df['diagnosis'] == 'neurofibroma') | (bcn_20000_df['diagnosis'] == 'scar') | (bcn_20000_df['diagnosis'] == 'verucca') |
                            (bcn_20000_df['diagnosis'] == 'acrochordon') | (bcn_20000_df['diagnosis'] == 'angiofibroma') | (bcn_20000_df['diagnosis'] =='fibrous papule')| 
                            (bcn_20000_df['diagnosis'] == 'cafe-au-lait macule') | (bcn_20000_df['diagnosis'] == 'angiokeratoma') | (bcn_20000_df['diagnosis'] == 'clear cell acanthoma')|
                            (bcn_20000_df['diagnosis'] =='melanoma') | (bcn_20000_df['diagnosis'] == 'basal cell carcinoma') | (bcn_20000_df['diagnosis'] == 'actinic keratosis') | 
                            (bcn_20000_df['diagnosis'] == 'squamous cell carcinoma') | (bcn_20000_df['diagnosis'] =='melanoma metastasis') | 
                            (bcn_20000_df['diagnosis'] == 'atypical melanocytic proliferation') | (bcn_20000_df['diagnosis'] == 'atypical spitz tumor')]

challenge_2016_test_df = challenge_2016_test_df[(challenge_2016_test_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2016_test_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2016_test_df['diagnosis'] == 'nevus') | (challenge_2016_test_df['diagnosis'] == 'solar lentigo') | (challenge_2016_test_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2016_test_df['diagnosis'] == 'vascular lesion') | (challenge_2016_test_df['diagnosis'] == 'lentigo nos') | (challenge_2016_test_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2016_test_df['diagnosis'] == 'lentigo simplex') | (challenge_2016_test_df['diagnosis'] == 'aimp') | (challenge_2016_test_df['diagnosis'] == 'angioma') |
                            (challenge_2016_test_df['diagnosis'] == 'neurofibroma') | (challenge_2016_test_df['diagnosis'] == 'scar') | (challenge_2016_test_df['diagnosis'] == 'verucca') |
                            (challenge_2016_test_df['diagnosis'] == 'acrochordon') | (challenge_2016_test_df['diagnosis'] == 'angiofibroma') | (challenge_2016_test_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2016_test_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2016_test_df['diagnosis'] == 'angiokeratoma') | (challenge_2016_test_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2016_test_df['diagnosis'] =='melanoma') | (challenge_2016_test_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2016_test_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2016_test_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2016_test_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2016_test_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2016_test_df['diagnosis'] == 'atypical spitz tumor')]

challenge_2016_train_df = challenge_2016_train_df[(challenge_2016_train_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2016_train_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2016_train_df['diagnosis'] == 'nevus') | (challenge_2016_train_df['diagnosis'] == 'solar lentigo') | (challenge_2016_train_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2016_train_df['diagnosis'] == 'vascular lesion') | (challenge_2016_train_df['diagnosis'] == 'lentigo nos') | (challenge_2016_train_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2016_train_df['diagnosis'] == 'lentigo simplex') | (challenge_2016_train_df['diagnosis'] == 'aimp') | (challenge_2016_train_df['diagnosis'] == 'angioma') |
                            (challenge_2016_train_df['diagnosis'] == 'neurofibroma') | (challenge_2016_train_df['diagnosis'] == 'scar') | (challenge_2016_train_df['diagnosis'] == 'verucca') |
                            (challenge_2016_train_df['diagnosis'] == 'acrochordon') | (challenge_2016_train_df['diagnosis'] == 'angiofibroma') | (challenge_2016_train_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2016_train_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2016_train_df['diagnosis'] == 'angiokeratoma') | (challenge_2016_train_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2016_train_df['diagnosis'] =='melanoma') | (challenge_2016_train_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2016_train_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2016_train_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2016_train_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2016_train_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2016_train_df['diagnosis'] == 'atypical spitz tumor')]


challenge_2017_test_df = challenge_2017_test_df[(challenge_2017_test_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2017_test_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2017_test_df['diagnosis'] == 'nevus') | (challenge_2017_test_df['diagnosis'] == 'solar lentigo') | (challenge_2017_test_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2017_test_df['diagnosis'] == 'vascular lesion') | (challenge_2017_test_df['diagnosis'] == 'lentigo nos') | (challenge_2017_test_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2017_test_df['diagnosis'] == 'lentigo simplex') | (challenge_2017_test_df['diagnosis'] == 'aimp') | (challenge_2017_test_df['diagnosis'] == 'angioma') |
                            (challenge_2017_test_df['diagnosis'] == 'neurofibroma') | (challenge_2017_test_df['diagnosis'] == 'scar') | (challenge_2017_test_df['diagnosis'] == 'verucca') |
                            (challenge_2017_test_df['diagnosis'] == 'acrochordon') | (challenge_2017_test_df['diagnosis'] == 'angiofibroma') | (challenge_2017_test_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2017_test_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2017_test_df['diagnosis'] == 'angiokeratoma') | (challenge_2017_test_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2017_test_df['diagnosis'] =='melanoma') | (challenge_2017_test_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2017_test_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2017_test_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2017_test_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2017_test_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2017_test_df['diagnosis'] == 'atypical spitz tumor')]


challenge_2017_training_df = challenge_2017_training_df[(challenge_2017_training_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2017_training_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2017_training_df['diagnosis'] == 'nevus') | (challenge_2017_training_df['diagnosis'] == 'solar lentigo') | (challenge_2017_training_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2017_training_df['diagnosis'] == 'vascular lesion') | (challenge_2017_training_df['diagnosis'] == 'lentigo nos') | (challenge_2017_training_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2017_training_df['diagnosis'] == 'lentigo simplex') | (challenge_2017_training_df['diagnosis'] == 'aimp') | (challenge_2017_training_df['diagnosis'] == 'angioma') |
                            (challenge_2017_training_df['diagnosis'] == 'neurofibroma') | (challenge_2017_training_df['diagnosis'] == 'scar') | (challenge_2017_training_df['diagnosis'] == 'verucca') |
                            (challenge_2017_training_df['diagnosis'] == 'acrochordon') | (challenge_2017_training_df['diagnosis'] == 'angiofibroma') | (challenge_2017_training_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2017_training_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2017_training_df['diagnosis'] == 'angiokeratoma') | (challenge_2017_training_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2017_training_df['diagnosis'] =='melanoma') | (challenge_2017_training_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2017_training_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2017_training_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2017_training_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2017_training_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2017_training_df['diagnosis'] == 'atypical spitz tumor')]


challenge_2017_validation_df = challenge_2017_validation_df[(challenge_2017_validation_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2017_validation_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2017_validation_df['diagnosis'] == 'nevus') | (challenge_2017_validation_df['diagnosis'] == 'solar lentigo') | (challenge_2017_validation_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2017_validation_df['diagnosis'] == 'vascular lesion') | (challenge_2017_validation_df['diagnosis'] == 'lentigo nos') | (challenge_2017_validation_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2017_validation_df['diagnosis'] == 'lentigo simplex') | (challenge_2017_validation_df['diagnosis'] == 'aimp') | (challenge_2017_validation_df['diagnosis'] == 'angioma') |
                            (challenge_2017_validation_df['diagnosis'] == 'neurofibroma') | (challenge_2017_validation_df['diagnosis'] == 'scar') | (challenge_2017_validation_df['diagnosis'] == 'verucca') |
                            (challenge_2017_validation_df['diagnosis'] == 'acrochordon') | (challenge_2017_validation_df['diagnosis'] == 'angiofibroma') | (challenge_2017_validation_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2017_validation_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2017_validation_df['diagnosis'] == 'angiokeratoma') | (challenge_2017_validation_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2017_validation_df['diagnosis'] =='melanoma') | (challenge_2017_validation_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2017_validation_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2017_validation_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2017_validation_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2017_validation_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2017_validation_df['diagnosis'] == 'atypical spitz tumor')]


challenge_2018_task_1_2_test_df = challenge_2018_task_1_2_test_df[(challenge_2018_task_1_2_test_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2018_task_1_2_test_df['diagnosis'] == 'nevus') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'solar lentigo') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2018_task_1_2_test_df['diagnosis'] == 'vascular lesion') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'lentigo nos') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2018_task_1_2_test_df['diagnosis'] == 'lentigo simplex') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'aimp') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'angioma') |
                            (challenge_2018_task_1_2_test_df['diagnosis'] == 'neurofibroma') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'scar') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'verucca') |
                            (challenge_2018_task_1_2_test_df['diagnosis'] == 'acrochordon') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'angiofibroma') | (challenge_2018_task_1_2_test_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2018_task_1_2_test_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'angiokeratoma') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2018_task_1_2_test_df['diagnosis'] =='melanoma') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2018_task_1_2_test_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2018_task_1_2_test_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2018_task_1_2_test_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2018_task_1_2_test_df['diagnosis'] == 'atypical spitz tumor')]


challenge_2018_task_1_2_training_df = challenge_2018_task_1_2_training_df[(challenge_2018_task_1_2_training_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2018_task_1_2_training_df['diagnosis'] == 'nevus') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'solar lentigo') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2018_task_1_2_training_df['diagnosis'] == 'vascular lesion') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'lentigo nos') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2018_task_1_2_training_df['diagnosis'] == 'lentigo simplex') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'aimp') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'angioma') |
                            (challenge_2018_task_1_2_training_df['diagnosis'] == 'neurofibroma') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'scar') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'verucca') |
                            (challenge_2018_task_1_2_training_df['diagnosis'] == 'acrochordon') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'angiofibroma') | (challenge_2018_task_1_2_training_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2018_task_1_2_training_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'angiokeratoma') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2018_task_1_2_training_df['diagnosis'] =='melanoma') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2018_task_1_2_training_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2018_task_1_2_training_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2018_task_1_2_training_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2018_task_1_2_training_df['diagnosis'] == 'atypical spitz tumor')]


challenge_2018_task_3_training_df = challenge_2018_task_3_training_df[(challenge_2018_task_3_training_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2018_task_3_training_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2018_task_3_training_df['diagnosis'] == 'nevus') | (challenge_2018_task_3_training_df['diagnosis'] == 'solar lentigo') | (challenge_2018_task_3_training_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2018_task_3_training_df['diagnosis'] == 'vascular lesion') | (challenge_2018_task_3_training_df['diagnosis'] == 'lentigo nos') | (challenge_2018_task_3_training_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2018_task_3_training_df['diagnosis'] == 'lentigo simplex') | (challenge_2018_task_3_training_df['diagnosis'] == 'aimp') | (challenge_2018_task_3_training_df['diagnosis'] == 'angioma') |
                            (challenge_2018_task_3_training_df['diagnosis'] == 'neurofibroma') | (challenge_2018_task_3_training_df['diagnosis'] == 'scar') | (challenge_2018_task_3_training_df['diagnosis'] == 'verucca') |
                            (challenge_2018_task_3_training_df['diagnosis'] == 'acrochordon') | (challenge_2018_task_3_training_df['diagnosis'] == 'angiofibroma') | (challenge_2018_task_3_training_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2018_task_3_training_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2018_task_3_training_df['diagnosis'] == 'angiokeratoma') | (challenge_2018_task_3_training_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2018_task_3_training_df['diagnosis'] =='melanoma') | (challenge_2018_task_3_training_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2018_task_3_training_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2018_task_3_training_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2018_task_3_training_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2018_task_3_training_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2018_task_3_training_df['diagnosis'] == 'atypical spitz tumor')]

challenge_2019_training_df = challenge_2019_training_df[(challenge_2019_training_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2019_training_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2019_training_df['diagnosis'] == 'nevus') | (challenge_2019_training_df['diagnosis'] == 'solar lentigo') | (challenge_2019_training_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2019_training_df['diagnosis'] == 'vascular lesion') | (challenge_2019_training_df['diagnosis'] == 'lentigo nos') | (challenge_2019_training_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2019_training_df['diagnosis'] == 'lentigo simplex') | (challenge_2019_training_df['diagnosis'] == 'aimp') | (challenge_2019_training_df['diagnosis'] == 'angioma') |
                            (challenge_2019_training_df['diagnosis'] == 'neurofibroma') | (challenge_2019_training_df['diagnosis'] == 'scar') | (challenge_2019_training_df['diagnosis'] == 'verucca') |
                            (challenge_2019_training_df['diagnosis'] == 'acrochordon') | (challenge_2019_training_df['diagnosis'] == 'angiofibroma') | (challenge_2019_training_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2019_training_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2019_training_df['diagnosis'] == 'angiokeratoma') | (challenge_2019_training_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2019_training_df['diagnosis'] =='melanoma') | (challenge_2019_training_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2019_training_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2019_training_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2019_training_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2019_training_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2019_training_df['diagnosis'] == 'atypical spitz tumor')]

challenge_2020_training_metadata_df = challenge_2020_training_metadata_df[(challenge_2020_training_metadata_df['diagnosis'] == 'seborrheic keratosis') | (challenge_2020_training_metadata_df['diagnosis'] == 'pigmented benign keratosis') |
                            (challenge_2020_training_metadata_df['diagnosis'] == 'nevus') | (challenge_2020_training_metadata_df['diagnosis'] == 'solar lentigo') | (challenge_2020_training_metadata_df['diagnosis'] == 'dermatofribroma') |
                            (challenge_2020_training_metadata_df['diagnosis'] == 'vascular lesion') | (challenge_2020_training_metadata_df['diagnosis'] == 'lentigo nos') | (challenge_2020_training_metadata_df['diagnosis'] == 'lichenoid keratosis') |
                            (challenge_2020_training_metadata_df['diagnosis'] == 'lentigo simplex') | (challenge_2020_training_metadata_df['diagnosis'] == 'aimp') | (challenge_2020_training_metadata_df['diagnosis'] == 'angioma') |
                            (challenge_2020_training_metadata_df['diagnosis'] == 'neurofibroma') | (challenge_2020_training_metadata_df['diagnosis'] == 'scar') | (challenge_2020_training_metadata_df['diagnosis'] == 'verucca') |
                            (challenge_2020_training_metadata_df['diagnosis'] == 'acrochordon') | (challenge_2020_training_metadata_df['diagnosis'] == 'angiofibroma') | (challenge_2020_training_metadata_df['diagnosis'] =='fibrous papule')| 
                            (challenge_2020_training_metadata_df['diagnosis'] == 'cafe-au-lait macule') | (challenge_2020_training_metadata_df['diagnosis'] == 'angiokeratoma') | (challenge_2020_training_metadata_df['diagnosis'] == 'clear cell acanthoma')|
                            (challenge_2020_training_metadata_df['diagnosis'] =='melanoma') | (challenge_2020_training_metadata_df['diagnosis'] == 'basal cell carcinoma') | (challenge_2020_training_metadata_df['diagnosis'] == 'actinic keratosis') | 
                            (challenge_2020_training_metadata_df['diagnosis'] == 'squamous cell carcinoma') | (challenge_2020_training_metadata_df['diagnosis'] =='melanoma metastasis') | 
                            (challenge_2020_training_metadata_df['diagnosis'] == 'atypical melanocytic proliferation') | (challenge_2020_training_metadata_df['diagnosis'] == 'atypical spitz tumor')]

ham10000_metadata_2023_df = ham10000_metadata_2023_df[(ham10000_metadata_2023_df['diagnosis'] == 'seborrheic keratosis') | (ham10000_metadata_2023_df['diagnosis'] == 'pigmented benign keratosis') |
                            (ham10000_metadata_2023_df['diagnosis'] == 'nevus') | (ham10000_metadata_2023_df['diagnosis'] == 'solar lentigo') | (ham10000_metadata_2023_df['diagnosis'] == 'dermatofribroma') |
                            (ham10000_metadata_2023_df['diagnosis'] == 'vascular lesion') | (ham10000_metadata_2023_df['diagnosis'] == 'lentigo nos') | (ham10000_metadata_2023_df['diagnosis'] == 'lichenoid keratosis') |
                            (ham10000_metadata_2023_df['diagnosis'] == 'lentigo simplex') | (ham10000_metadata_2023_df['diagnosis'] == 'aimp') | (ham10000_metadata_2023_df['diagnosis'] == 'angioma') |
                            (ham10000_metadata_2023_df['diagnosis'] == 'neurofibroma') | (ham10000_metadata_2023_df['diagnosis'] == 'scar') | (ham10000_metadata_2023_df['diagnosis'] == 'verucca') |
                            (ham10000_metadata_2023_df['diagnosis'] == 'acrochordon') | (ham10000_metadata_2023_df['diagnosis'] == 'angiofibroma') | (ham10000_metadata_2023_df['diagnosis'] =='fibrous papule')| 
                            (ham10000_metadata_2023_df['diagnosis'] == 'cafe-au-lait macule') | (ham10000_metadata_2023_df['diagnosis'] == 'angiokeratoma') | (ham10000_metadata_2023_df['diagnosis'] == 'clear cell acanthoma')|
                            (ham10000_metadata_2023_df['diagnosis'] =='melanoma') | (ham10000_metadata_2023_df['diagnosis'] == 'basal cell carcinoma') | (ham10000_metadata_2023_df['diagnosis'] == 'actinic keratosis') | 
                            (ham10000_metadata_2023_df['diagnosis'] == 'squamous cell carcinoma') | (ham10000_metadata_2023_df['diagnosis'] =='melanoma metastasis') | 
                            (ham10000_metadata_2023_df['diagnosis'] == 'atypical melanocytic proliferation') | (ham10000_metadata_2023_df['diagnosis'] == 'atypical spitz tumor')]

prove_ai_metadata_2023_df = prove_ai_metadata_2023_df[(prove_ai_metadata_2023_df['diagnosis'] == 'seborrheic keratosis') | (prove_ai_metadata_2023_df['diagnosis'] == 'pigmented benign keratosis') |
                            (prove_ai_metadata_2023_df['diagnosis'] == 'nevus') | (prove_ai_metadata_2023_df['diagnosis'] == 'solar lentigo') | (prove_ai_metadata_2023_df['diagnosis'] == 'dermatofribroma') |
                            (prove_ai_metadata_2023_df['diagnosis'] == 'vascular lesion') | (prove_ai_metadata_2023_df['diagnosis'] == 'lentigo nos') | (prove_ai_metadata_2023_df['diagnosis'] == 'lichenoid keratosis') |
                            (prove_ai_metadata_2023_df['diagnosis'] == 'lentigo simplex') | (prove_ai_metadata_2023_df['diagnosis'] == 'aimp') | (prove_ai_metadata_2023_df['diagnosis'] == 'angioma') |
                            (prove_ai_metadata_2023_df['diagnosis'] == 'neurofibroma') | (prove_ai_metadata_2023_df['diagnosis'] == 'scar') | (prove_ai_metadata_2023_df['diagnosis'] == 'verucca') |
                            (prove_ai_metadata_2023_df['diagnosis'] == 'acrochordon') | (prove_ai_metadata_2023_df['diagnosis'] == 'angiofibroma') | (prove_ai_metadata_2023_df['diagnosis'] =='fibrous papule')| 
                            (prove_ai_metadata_2023_df['diagnosis'] == 'cafe-au-lait macule') | (prove_ai_metadata_2023_df['diagnosis'] == 'angiokeratoma') | (prove_ai_metadata_2023_df['diagnosis'] == 'clear cell acanthoma')|
                            (prove_ai_metadata_2023_df['diagnosis'] =='melanoma') | (prove_ai_metadata_2023_df['diagnosis'] == 'basal cell carcinoma') | (prove_ai_metadata_2023_df['diagnosis'] == 'actinic keratosis') | 
                            (prove_ai_metadata_2023_df['diagnosis'] == 'squamous cell carcinoma') | (prove_ai_metadata_2023_df['diagnosis'] =='melanoma metastasis') | 
                            (prove_ai_metadata_2023_df['diagnosis'] == 'atypical melanocytic proliferation') | (prove_ai_metadata_2023_df['diagnosis'] == 'atypical spitz tumor')]


file_paths = ['/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/BCN_2020_data/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/HAM_10000/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2016_data/2016_test_data/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2016_data/2016_training_data/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2017_data/2017_test/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2017_data/2017_training/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2017_data/2017_validation/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2018_data/2018_task_1-2_test/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2018_data/2018_task_1-2_validation/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2018_data/2018_task_3_training/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2019_data/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/2020_data/training_data/archive/',
              '/home/steve/Desktop/skin_cancer_detector/Skin_Cancer_NN_Project_Images/PROVe-AI/archive/' ]

changed_file_name_prefixes = ['BCN_20000','HAM_10000','2016_test','2016_train','2017_test','2017_train', '2017_validation', '2018_task_1-2_test',
                              '2018_task_1-2-validation','2018_task_3_training', '2019_data', '2020_data_training','PROVe-AI']

destination_folder = '/home/steve/Desktop/skin_cancer_detector/full_training_data/'

benigns = 0
malignants = 0

for df_zipper in zip( [bcn_20000_df, ham10000_metadata_2023_df, challenge_2016_test_df, challenge_2016_train_df, challenge_2017_test_df,challenge_2017_training_df,challenge_2017_validation_df,
           challenge_2018_task_1_2_test_df, challenge_2018_task_1_2_validation_df, challenge_2018_task_3_training_df, challenge_2019_training_df,
           challenge_2020_training_metadata_df, prove_ai_metadata_2023_df], file_paths, changed_file_name_prefixes):
    
    zipper_df = df_zipper[0].reset_index()

    #print('my name is')
    
    zipper_df['changed_filenames'] = ['x' for i in range(zipper_df.shape[0])]
    
    for i in range(zipper_df.shape[0]):

        zipper_df['changed_filenames'][i] =  df_zipper[2] + '_' + zipper_df['isic_id'][i]

    for ind in zipper_df.index:

        photo_path =  df_zipper[1] +  zipper_df['isic_id'][ind] + '.JPG'
        img = load_img(photo_path)
        img = tf.image.resize(img,(299,299), method='bilinear')
        benign_or_malignant = ""

        if zipper_df['diagnosis'][ind] in BENIGN_TYPES:
            benign_or_malignant = 'benign'
            benigns += 1

        elif zipper_df['diagnosis'][ind] in MALIGNANT_TYPES:
            benign_or_malignant = 'malignant'
            malignants += 1

        else:
            print('choke!')
            print(zipper_df['diagnosis'][ind])
            sys.exit(1)

        destination = destination_folder + benign_or_malignant + '/'+ str(zipper_df['changed_filenames'][ind]) + '.JPEG'
        save_img(destination, img, scale=True)


print("benigns ", benigns)
print("malignants, ", malignants)
print("end program")
    

