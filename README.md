# Avito demand prediction 

when a seller place an Ad on the platform, try to predict it review/selling probability for that item. We got text/image/tabular data.

# workflow 

## eda 
- `eda1.ipynb` explore
- supplement?? 
    - same ad display periods as train (3/15-3/28) 
    - how to use it?
- `item_id` is pkey (train/test)
- `user_id` map to `uidx`, smaller `uidx`(almost) imply larger # of `item_seq_number` per user (higher corr with prob)

## features engineering
- `feat0_basic0.ipynb`: 
    - `uidx`, `iidx`, `iid` : replace `user_id`,`item_seq_number`,`item_id`
    - `region_city_label` encode combination of `region`, `city`
- high corr vs `deal_probabilty`
    - `image_top_1` 
    - `uidx`
    - `ads_cnt_by_iid`

- (from) model importance 
- FM 
     - `deal_prob` by (`uidx`,`iidx`,`region_city2_label`,)
        - all latents vector ~ zeros 
        - only bias (some have high corr with `deal_prob` )
        - interaction(poly2) importance may not exist
     - train on groupby count 
        - cnt by (`uidx`,`region_city_label`)
        - save to `fm_uidx_rc_cnt_lat2.h5`
- images 
- text 
    work on `description`, and `title` columns
    - tfidf (ngram=2,max_feats=10**5) + tsvd(dim=5)
    - hashing (ngram=2) + tsvd(dim=5) 
## cv 
    - 
## modeling 
    - lgbm
    - FM
    - nn 

## ensemble 
    - blending 
    - 

# MySubmit History
1. try basic_feat0 + text_feat0 +  in lgbm
# useful resource
kernel, discussion, paper ...etc
- trick
    - [tips and trick working with large dataset](https://www.kaggle.com/frankherfert/tips-tricks-for-working-with-large-datasets/code)
-  image
    - [recognition](https://www.kaggle.com/wesamelshamy/ad-image-recognition-and-quality-scoring/code)
        - [keras pretrain model](https://www.kaggle.com/gaborfodor/keras-pretrained-models)
    - [quality](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality)
    - image features 
        - [Natural Growth Patterns (Fractals of Nature)](https://www.kaggle.com/the1owl/natural-growth-patterns-fractals-of-nature/code)
        - 
    - image caption 
        - [tutorial](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
- text
    - [using train_active for word embedding](https://www.kaggle.com/christofhenkel/using-train-active-for-training-word-embeddings/code)

    - 
- how to do Factorization Machine ?
    - origin paper about FM [Rendle2010FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    - python implement by top kaggler CPMP [here](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Implementing_Libfm_in_Keras?lang=en_us)
    