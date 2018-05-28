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
    - `region_city_label`
    - `param1`,`param2`,`parm3`
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
- mean encode 
    -[Encoding CV split](https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features)
    - useless xxxx

- zero prediction (meta_model) 
    - no help (why?)
## cv
- `lgb.cv`: (fold=4)
    - params:       

            lgbm_params =  {
                                'task': 'train',
                                'boosting_type': 'gbdt',
                                'objective': 'regression',
                                'metric': 'rmse',    
                                'max_depth': -1,
                                'num_leaves': 33,
                                'feature_fraction': 0.7,
                                'bagging_fraction': 0.8,                                
                                'learning_rate': 0.1,
                                'verbose': 20
                            },
            categorical =[
               'month','day','weekday',
               'param_1',
               'param_2','param_3',
               'category_name','parent_category_name',
               'region_city_label', 
               'user_type'
            ]
                        
        
    - with feat0 18 featurs :
        - base `feat0` :`0.225915 + 0.000336991` --> (region_city_label as categorical)  `0.225538 + 0.000292427`
        - add `cnt_by_uidx_param1`: `0.2256xxx + 0.0003...`
    - with `feat2_trn_inter_svd` 
        - (add 5*2 +1 feats)
            (`uidx x param_1`, `uidx x iidx`, `cnt_by_uidx_param1`)
            -  `0.22572 + 0.000345192`
        
    - with text (5*4) features         
        - modify (with `cleanName` func --textfeats) : `0.223908 + 0.000289244` --> `0.223626 + 0.000272922`
        - add `fm_uidx_rc_cnt_latent2` (+5 feats): `0.22397 + 0.000289225` (no help!!)
        - add `feat2_trn_inter_svd` (+11 feats): `0.223852 + 0.000292917` (not that help !!)
    - with zero perdict (1) 
        - no help (cutoff=0.8,T/F) --> `0.225566 + 0.000314014` wo cutoff -->`0.2258`
    - with mean encode by  
        - no help --> `0.226273 + 0.000289295`
        - replace category by mean_target --> `0.222957 + 0.000236522`
        
## modeling 
    - lgbm
    - FM
    - nn 

## ensemble 
    - blending 
    - stacking
        - [starter](https://www.kaggle.com/mmueller/stacking-starter)

# MySubmit History

1. local cv:  0.2216 (lb: 0.2276 overfit?)
    - text(20) + mean_target + basic
2. local cv:  0.2238 (lb: 0.2272)
    - text + basic
3. local cv:  0.223 (lb: 0.2264)
    - modify image_top_1 to cat
4. local cv:  0.2178 (lb: 0.2268 overfit)
    - (text+basic) + mean (nfolds=100)
5. local cv:  0.2201 (lb: 0.2273 overfit...)
    - modify mean with same out of fold idx
6. local cv:  0.2213 (lb: 0.2280 overfit)
    - mean without strified kfolds (use only kfold=4)
7. local cv:  0.2231 (lb: 0.2274 )
    - mean encode cv2 (only used)
        - 'mean_region_city_label',
        - 'mean_user_type', 
        - 'mean_parent_category_name', 
        - 'mean_category_name',
        - 'mean_image_top_1
8. local cv: 0.1647 (lb: 0.2383)
# useful resource
kernel, discussion, paper ...etc
- trick
    - [tips and trick working with large dataset](https://www.kaggle.com/frankherfert/tips-tricks-for-working-with-large-datasets/code)
-  image
    - quality
        - [recognition](https://www.kaggle.com/wesamelshamy/ad-image-recognition-and-quality-scoring/code)
        - [keras pretrain model](https://www.kaggle.com/gaborfodor/keras-pretrained-models)
        - [image quality](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality)
        - [Natural Growth Patterns (Fractals of Nature)](https://www.kaggle.com/the1owl/natural-growth-patterns-fractals-of-nature/code)
        
    - mp
        - [idea for image features multiprocessing support](https://www.kaggle.com/liuhdsgoal/ideas-for-image-features-multiprocessing-support)
    - image caption 
        - [tutorial](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
- text
    - [using train_active for word embedding](https://www.kaggle.com/christofhenkel/using-train-active-for-training-word-embeddings/code)

    - adjust tsvd + tfidf dissucssion [here](https://www.kaggle.com/c/avito-demand-prediction/discussion/56798)
    - [Feature union and ridge trick](https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code)
- how to do Factorization Machine ?
    - origin paper about FM [Rendle2010FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    - python implement by top kaggler CPMP [here](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Implementing_Libfm_in_Keras?lang=en_us)
    