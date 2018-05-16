# Avito demand prediction 

when a seller place an Ad on the platform, try to predict it review/selling probability for that item. We got text/image/tabular data.
# workflow 
## eda 
- supplement?? 
    - same ad display periods as train (3/15-3/28) 
    - how to use it?
- 
- `item_id` is pkey (train/test)
- `user_id` map to `uidx`, smaller uidx larger # of item_id per user (higher corr prob)
- 
## features 

- high corr vs `deal_probabilty`
    - `image_top_1` 
    - `uidx`
    - `ads_cnt_by_iid`
    - 
- (from) model importance 
## cv 
    - 
## modeling 
    - lgbm
    - 

## ensemble 
    - blending 
    - 
# useful resource
kernel, discussion ...etc
- trick
    - [tips and trick working with large dataset](https://www.kaggle.com/frankherfert/tips-tricks-for-working-with-large-datasets/code)
-  image
    - [recognition](https://www.kaggle.com/wesamelshamy/ad-image-recognition-and-quality-scoring/code)
        - [keras pretrain model](https://www.kaggle.com/gaborfodor/keras-pretrained-models)
    - [quality](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality)

- text
    - [using train_active for word embedding](https://www.kaggle.com/christofhenkel/using-train-active-for-training-word-embeddings/code)

    -
- how to do Factorization Machine ?
    - [kaggler](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Implementing_Libfm_in_Keras?lang=en_us)