# %%
import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.lstur import LSTURModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))


# %%
epochs = 50
seed = 42
batch_size = 32

MIND_type = 'demo'

data_path = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo" # tmpdir.name

train_news_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/train/news.tsv" # 
train_behaviors_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/train/preprocess/preproc_Obf_0.01_9369Items_1000Users_behavior.tsv" # 
valid_news_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/valid/news.tsv" # 
valid_behaviors_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/valid/behaviors.tsv" # 
wordEmb_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/embedding.npy" # 
userDict_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/uid2index.pkl" # 
wordDict_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/word_dict.pkl" # 
yaml_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/lstur.yaml" # 

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs)
print(hparams)

iterator = MINDIterator

model = LSTURModel(hparams, iterator, seed=seed)

print ("Start fitting model")
model.fit(train_news_file, train_behaviors_file,valid_news_file, valid_behaviors_file)

res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
print(res_syn)

sb.glue("res_1%_Obf-1000Users-epoch50-lstur", res_syn)

model_path = os.path.join(data_path, "model_1%_Obf-1000Users-epoch50-lstur")#_1%Obf
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "LSTUR_ckpt_1%_Obf-1000Users-epoch50"))

group_impr_indexes, group_labels, group_preds = model.run_fast_eval(valid_news_file, valid_behaviors_file)

with open(os.path.join(data_path, 'LSTUR_predictions/res_1%_Obf-1000Users/prediction_1000_users_epochs50_1%_Obf.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank])+ '\n')

data_path= '/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/LSTUR_predictions/res_1%_Obf-1000Users/'
f = zipfile.ZipFile(os.path.join(data_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(data_path, 'prediction_1000_users_epochs50_1%_Obf.txt'), arcname='prediction_1000_users_epochs50_1%_Obf.txt')
f.close()