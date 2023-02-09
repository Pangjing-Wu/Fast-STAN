import argparse
import gc
import glob
import logging
import multiprocessing as mp
import os
import pickle

import numpy as np
import pandas as pd

from stan import FastSTAN


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='a'
    )
logger = logging.getLogger('stan')


def parse_args():
    parser = argparse.ArgumentParser(description='Sequence and Time Aware Neighborhood (STAN)')
    parser.add_argument('--train', required=True, help='training data path')
    parser.add_argument('--test', required=True, help='test data path')
    parser.add_argument('--ref', type=str, help='coarsely ranking results')
    parser.add_argument('--save', '-o', required=True, help='save path')
    parser.add_argument('--chunk', default=1000, help='test chunk number')
    parser.add_argument('--n_jobs', '-j', type=int, default=1, help='n paralleled processes')
    return parser.parse_args()


def load_session(path):
    file = glob.glob(path)
    data = [pd.read_parquet(f, engine='fastparquet') for f in file]
    data = pd.concat(data, axis=0)
    data['ts'] = data['ts'] // 1000
    return data


def list2str(arr, sep=' '):
    return sep.join(map(str, arr))


def mp_predict(model, data, reference, save_dir):
    f = open(save_dir, 'a')
    sessions  = data.groupby('session')
    for session, df in sessions:
        for type_ in ['clicks', 'carts', 'orders']:
            if isinstance(reference, dict):
                ref = reference[f"{session}_{type_}"]
            else:
                ref = reference
            item = model.predict_next(session=df, reference=ref)
            item = item.sort_values(ascending=False).iloc[:20].index.tolist()
            f.write(f'{session}_{type_},{list2str(item)}\n')
    f.close()


def main(args, stan_config):
    # load training set.
    logger.info('loading training files.')
    train_data = load_session(args.train)
    all_items  = train_data.aid.unique()
    logger.info(f'loaded {train_data.shape[0]} events.')
    
    # caching training sessions.
    logger.info(f'caching training sessions.')
    model = FastSTAN(**stan_config)
    model.fit(train_data)

    # load test set.
    logger.info('loading test files.')
    test_data = load_session(args.test)
    logger.info(f'loaded {test_data.shape[0]} events.')

    # load coarsely ranking results, format: {'[sessionID]_[type]': np.array}.
    if args.ref:
        logger.info(f'loading coarse ranking from "{args.ref}".')
        f = open(args.ref, 'rb')
        reference = pickle.load(f)
        f.close()
        logger.info(f'loaded {len(reference)} reference.')
    else:
        reference = all_items

    # create batches for multi-processing.
    logger.info(f'creating {args.chunk} test data chunks.')
    sessions   = test_data.session.unique()
    chunk_size = int(np.ceil(len(sessions) / args.chunk))
    chunks     = [test_data.loc[test_data.session.isin(sessions[i * chunk_size : (i+1) * chunk_size])] for i in range(args.chunk - 1)]
    chunks     = chunks + [test_data.loc[test_data.session.isin(sessions[(args.chunk - 1) * chunk_size:])]]
    chunks     = iter(chunks)
    del train_data, test_data
    gc.collect()

    # clean up saving directory and init.
    files = glob.glob(f'{args.save}.temp*')
    for file in files: 
        os.remove(file)
    files = [f'{args.save}.temp{pid}' for pid in range(args.n_jobs)]
    for file in files:
        f = open(file, 'w')
        f.write('session_type,labels\n')
        f.close()

    logger.info(f'predicting next items for each session with {args.n_jobs} subprocesses.')
    for i, chunk in enumerate(chunks):
        sessions   = chunk.session.unique()
        batch_size = int(np.ceil(len(sessions) / args.n_jobs))
        batches    = [chunk.loc[chunk.session.isin(sessions[i * batch_size : (i+1) * batch_size])] for i in range(args.n_jobs - 1)]
        batches    = batches + [chunk.loc[chunk.session.isin(sessions[(args.n_jobs - 1) * batch_size:])]]
        batches    = iter(batches)
        pool = list()
        for batch, file in zip(batches, files):
            kwargs = dict(model=model, data=batch, reference=reference, save_dir=file)
            pool.append(mp.Process(target=mp_predict, kwargs=kwargs))
        for p in pool: p.start()
        for p in pool: p.join()
        if i % (args.chunk // 100) == 0:
            logger.info(f'completed {i * 100 // args.chunk}% process.')

    logger.info(f'combining results.')
    data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    data.to_csv(args.save, index=False)
    for file in files: 
        os.remove(file)


if __name__ == '__main__':
    args = parse_args()
    stan_config = dict(
        k=1500,
        sample_size=2500,
        lambda_spw=0.905, 
        lambda_snh=10000,
        lambda_inh=0.4525,
        session_key='session',
        item_key='aid',
        time_key='ts'
        )
    main(args, stan_config=stan_config)
