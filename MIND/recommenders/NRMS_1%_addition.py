# # NRMS: Neural News Recommendation with Multi-Head Self-Attention
# NRMS \[1\] is a neural news recommendation approach with multi-head selfattention. The core of NRMS is a news encoder and a user encoder. In the newsencoder, a multi-head self-attentions is used to learn news representations from news titles by modeling the interactions between words. In the user encoder, we learn representations of users from their browsed news and use multihead self-attention to capture the relatedness between the news. Besides, we apply additive
# attention to learn more informative news and user representations by selecting important words and news.
# 
# ## Properties of NRMS:
# - NRMS is a content-based neural news recommendation approach.
# - It uses multi-self attention to learn news representations by modeling the iteractions between words and learn user representations by capturing the relationship between user browsed news.
# - NRMS uses additive attentions to learn informative news and user representations by selecting important words and news.
# 
# ## Data format:
# For quicker training and evaluaiton, we sample MINDdemo dataset of 5k users from [MIND small dataset](https://msnews.github.io/). The MINDdemo dataset has the same file format as MINDsmall and MINDlarge. If you want to try experiments on MINDsmall and MINDlarge, please change the dowload source. Select the MIND_type parameter from ['large', 'small', 'demo'] to choose dataset.

# %%
# !pip3 install recommenders

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
from recommenders.models.newsrec.models.nrms import NRMSModel
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

train_news_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/train/news.tsv" # os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/train/preprocess/preproc_Add_0.01_9369Items_1000Users_behavior.tsv" # 
valid_news_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/valid/news.tsv" # os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/valid/behaviors.tsv" # os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/embedding_all.npy" # os.path.join(data_path, "utils", "embedding_all.npy")
userDict_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/uid2index.pkl" # os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/word_dict_all.pkl" # os.path.join(data_path, "utils", "word_dict_all.pkl")
vertDict_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/vert_dict.pkl" # os.path.join(data_path, "utils", "vert_dict.pkl")
subvertDict_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/subvert_dict.pkl" # os.path.join(data_path, "utils", "subvert_dict.pkl")
yaml_file = "/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/utils/nrms.yaml" # os.path.join(data_path, "utils", r'naml.yaml')


hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams)


iterator = MINDIterator

model = NRMSModel(hparams, iterator, seed=seed)

print ("Start fitting model")
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
print(res_syn)

sb.glue("res_1%_Add-1000Users-epoch50", res_syn)

model_path = os.path.join(data_path, "model_1%_Add-1000Users-epoch50")#_1%Obf
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "nrms_ckpt_1%_Add-1000Users-epoch50"))

group_impr_indexes, group_labels, group_preds = model.run_fast_eval(valid_news_file, valid_behaviors_file)

with open(os.path.join(data_path, 'NRMS_predictions/res_1%_Add-1000Users/prediction_1000_users_epochs50_1%_Add.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank])+ '\n')

data_path= '/export/scratch2/home/manel/RecSys_News/MIND/Version_Mind_small-Demo/NRMS_predictions/res_1%_Add-1000Users/'
f = zipfile.ZipFile(os.path.join(data_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(data_path, 'prediction_1000_users_epochs50_1%_Add.txt'), arcname='prediction_1000_users_epochs50_1%_Add.txt')
f.close()