# %%
import pandas as pd
import numpy as np
import yaml
from src.configs.data_config import TIME_VARYING_UNKNOWN_REALS,TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_KNOWN_REALS,TIME_VARYING_KNOWN_CATEGORICALS,STATIC_REALS,STATIC_CATEGORICALS,TARGET
from src.configs.data_config import TF_RECORDS_PER_SHARD
from src.data.tf_data_utils import TimeseriesDataTransformer,TFdata_serializer
import tensorflow as tf
import joblib
import os

def load_data():
    """Load raw data including list of uids for training validation and test"""

    data  = pd.read_csv('data/interim/raw.csv',index_col=0)

    with open('data/interim/train_val_test_uids.yml', 'r') as file:
        uids = yaml.safe_load(file)

    return data, uids

def build_output_dirs(paths):
    "Function to build output directories for saved data"

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


# %%
if __name__ == '__main__':

    #Create new output directory for TF records
    new_dirs = [f'./data/processed/TF_records/{subset}/' for subset in ['train','val','test']]
    new_dirs+= ['./models/']
    build_output_dirs(new_dirs)

    # %%

    #Load the data and uids
    data, uids = load_data()

    #Defien data transformer for pre-processing data
    transformer = TimeseriesDataTransformer(time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
                                    time_varying_unknown_categoricals=TIME_VARYING_UNKNOWN_CATEGORICALS,
                                    time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
                                    time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
                                    static_reals=STATIC_REALS,
                                    static_categoricals=STATIC_CATEGORICALS)

    #Fit and save the transformer
    transformer.fit(data[lambda x:x.uid.isin(uids['train'])])
    joblib.dump(transformer,'./models/ts_data_transformer.joblib')

    idx_dict = {'train':[],'val':[],'test':[],}

    #Loop to output TF records
    for subset in ['train','val','test']:

        subset_uids = uids[subset] 

        #Calculate number of required shards
        n_shards = (len(subset_uids)//TF_RECORDS_PER_SHARD)+(1 if len(subset_uids) % TF_RECORDS_PER_SHARD != 0 else 0)
        
        index=0 #stard index

        for shard in range(n_shards):
            
            filename=f"./data/processed/TF_records/{subset}/shard_{shard}.tfrecord"
            end = index + TF_RECORDS_PER_SHARD if len(subset_uids) > (index + TF_RECORDS_PER_SHARD) else len(subset_uids)
            
            with tf.io.TFRecordWriter(filename) as writer:

                for uid in subset_uids[index:end]:

                    player_data = data[lambda x:x.uid==uid] #subset player
                    player_idx =player_data.index.values.tolist()
                    idx_dict[subset]+=player_idx

                    #Transform the data
                    static,time_varying_know,time_varying_unknow = transformer.transform(player_data)
                    
                    #Serialize the data and add meta info
                    se = TFdata_serializer().serialize_element(index=player_data.index.values,
                                                                static_features=static,
                                                                time_varying_known_features=time_varying_know,
                                                                time_varying_unknown_features=time_varying_unknow,
                                                                labels=player_data[TARGET].values)

                    #Write to record
                    writer.write(se.SerializeToString())

                #Update index
                index=end


    # As the dl data is sorted by player id (not the original index) we can get the
    # sorted index of our player list for rearranging the test predictions
    dl_test_idx = np.array(idx_dict['test'])
    dl2ml_test_idx = np.argsort(dl_test_idx)
    np.save('./data/interim/dl2ml_test_idx.npy',dl2ml_test_idx)