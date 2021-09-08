# solution
Usage
Dependencies: python>=3.6, numpy, pandas, torch>=1.3, tqdm.

After downloading the data,  put them into the solution/resources folder.
We'll first need to filter some users with too few interactions and merge all features together, and this is accomplished by data_pre.py. 
Then we'll pretrain embeddings for every user and item by running run_pretrain_embeddings_kaggle.py :

$ python data_pre.py
$ python run_pretrain_embeddings_kaggle.py --lr 0.001 --n_epochs 4
You can tune the lr and n_epochs hyper-parameters to get better evaluate loss. Then we begin to train the model. Currently there are three algorithms in DBRL, so we can choose one of them:

$ python run_bcq_kaggle.py --n_epochs 5 --lr 1e-5
