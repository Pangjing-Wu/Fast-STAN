from operator import itemgetter
from math import sqrt, exp
import random

import numpy as np
import pandas as pd


class FastSTAN(object):
    '''
    Fast STAN accelerates training and inference process by:
    1. Multi-processing inference.
        According to my experimental analysis, the inference process in STAN takes much 
        more time than the fitting. It is necessary to parallelize the inference process 
        when conduct experiments on a large dataset. 
        Therefore, I forgo some not process-safe design (e.g. the incremental test data
        cache in inference) to enable inference the next item in a multi-process manner.
    2. Accelerate data operation.
        Making inference by pandas' methods is significantly faster than the build-in methods. 
    3. Support out-of-order input.
        Previous implement requires the input data ordered by time stamp. If the input data 
        is not ordered by session id, some data will missing. I release this require.

    Please note that multi-process inference can be only performed in an offline manner. 
    It sacrifices some online natures to achieve faster inference. If you would like to 
    perform online inference, please refer to:
    "https://github.com/rn5l/session-rec/blob/master/algorithms/knn/stan.py".

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 1500)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 5000)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    remind : bool
        Defines whether to update new item score. (default: True)
    lambda_spw : float
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (default: 1.02)
    lambda_snh : float
        Experimental function to give less weight to items from older sessions (default: 5.0)
    lambda_inh : float
        Experimental function to use the dwelling time for item view actions as a weight in the similarity calculation. (default: 2.05)
    session_key : string
        Header of the session ID column in the input file. (default: 'session')
    item_key : string
        Header of the item ID column in the input file. (default: 'item')
    time_key : string
        Header of the timestamp column in the input file. (default: 'time')
    '''

    def __init__(self, k, sample_size=5000, sampling='recent', remind=True, 
                 lambda_spw=1.02, lambda_snh=5.0, lambda_inh=2.05,
                 session_key='session', item_key='item', time_key='time'):
                 # {k: 1500, sample_size: 2500, lambda_spw: 0.905 , lambda_snh: 100, lambda_inh: 0.4525 }
        self.k = k
        self.sample_size = sample_size
        self.sampling    = sampling
        self.lambda_spw  = lambda_spw
        self.lambda_snh  = lambda_snh * 24 * 3600
        self.lambda_inh  = lambda_inh
        self.remind = remind
        # set keys.
        self.session_key = session_key
        self.item_key    = item_key
        self.time_key    = time_key
        # init.
        self.data = None
        self.session_item_map  = dict()
        self.session_last_time = dict()
        self.item_session_map  = dict()
        
    def fit(self, data:pd.DataFrame):
        '''
        Caching sessions' information.
        Parameters
        --------
        data: pandas.DataFram
            Training data. It contains the transactions of the sessions. 
            It has one column for session IDs, one for item IDs and one 
            for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must 
            correspond to the ones you set during the initialization of 
            the network (session_key, item_key, time_key properties).
        '''
        for session, df in data.groupby(self.session_key):
            df = df.sort_values(by=self.time_key, ascending=True).reindex()
            self.session_item_map.update({session: df[self.item_key].values})
            self.session_last_time.update({session: df[self.time_key].max()})
        for item, df in data.groupby(self.item_key):
            self.item_session_map.update({item: set(df[self.session_key])})
        
    def predict_next(self, session: pd.DataFrame, reference: np.ndarray):
        '''
        Gives predicton scores for a selected set of items on how likely 
        they be the next item in the session.
                
        Parameters
        --------
        session : pandas.DataFrame
            The session DataFrame with columns = ['sessionID', 'itemID', 'timestamp'].
        reference : 1D array
            IDs of items for which the network should give prediction scores. 
            Every ID must be in the set of item IDs of the training set.
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. 
            Indexed by the item IDs.
        
        '''
        reference = np.array(reference) if not isinstance(reference, np.ndarray) else reference
        session_items = session[self.item_key].values
        timestamp = int(session[self.time_key].iloc[-1])
        neighbors = self.find_neighbors(session_items, timestamp)
        scores = self.score_items(neighbors, session_items)
        predictions = np.zeros(len(reference))
        mask   = np.isin(reference, list(scores.keys()))
        scores = [scores[x] for x in reference[mask]]
        predictions[mask] = scores
        series = pd.Series(data=predictions, index=reference)
        return series
    
    def cosine(self, current, neighbor, pos_map):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        lneighbor = len(neighbor)
        intersection = current & neighbor
        if pos_map is not None:
            vp_sum = 0
            current_sum = 0
            for i in current:
                current_sum += pos_map[i] * pos_map[i]
                if i in intersection:
                    vp_sum += pos_map[i]
        else:
            vp_sum = len(intersection)
            current_sum = len(current)
        result = vp_sum / (sqrt(current_sum) * sqrt(lneighbor))
        return result
        
    def most_recent_sessions(self, sessions, number):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()
        tuples = list()
        for session in sessions:
            time = self.session_last_time[session]
            tuples.append((session, time))
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        return sample


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors(self, session_items, timestamp):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions(session_items)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors, timestamp)
        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]
        return possible_neighbors
    
    
    def possible_neighbor_sessions(self, session_items):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        Parameters
        --------
        session_items: set of item ids
        Returns 
        --------
        out : set           
        '''
        relevant_sessions = set()
        for item in session_items:
            relevant_sessions = relevant_sessions | self.item_session_map[item]
        # use all session as possible neighbors.
        if self.sample_size == 0:
            return relevant_sessions
        # sample some sessions.
        else:
            if len(relevant_sessions) > self.sample_size:
                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(relevant_sessions, self.sample_size)
                else:
                    sample = relevant_sessions[:self.sample_size]
                return sample
            else: 
                return relevant_sessions
                        

    def calc_similarity(self, session_items, sessions, timestamp):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        pos_map = None
        if self.lambda_spw:
            pos_map = {}
        length = len(session_items)
        pos = 1
        for item in session_items:
            if self.lambda_spw is not None: 
                pos_map[item] = self.session_pos_weight(pos, length, self.lambda_spw)
                pos += 1 
        items = set(session_items)
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            n_items = self.session_item_map[session]
            similarity = self.cosine(items, set(n_items), pos_map)          
            if self.lambda_snh is not None:
                sts   = self.session_last_time[session]
                decay = self.session_time_weight(timestamp, sts, self.lambda_snh)
                similarity *= decay          
            neighbors.append((session, similarity))
        return neighbors
    
    def session_pos_weight(self, position, length, lambda_spw):
        diff = position - length
        return exp(diff/lambda_spw)
    
    def session_time_weight(self, ts_current, ts_neighbor, lambda_snh):
        diff = ts_current - ts_neighbor
        return exp(-diff/lambda_snh)
            
    def score_items(self, neighbors, current_session):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        s_items = set(current_session)
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            n_items = self.session_item_map[session[0]]
            pos_last = {}
            pos_i_star = None
            for i in range(len(n_items)):
                if n_items[i] in s_items: 
                    pos_i_star = i + 1
                pos_last[n_items[i]] = i + 1
            n_items = set(n_items)
            for item in n_items:
                if not self.remind and item in s_items:
                    continue
                old_score = scores.get(item)
                new_score = session[1]
                if self.lambda_inh is not None:
                    new_score = new_score * self.item_pos_weight(pos_last[item], pos_i_star, self.lambda_inh)
                if not old_score is None:
                    new_score = old_score + new_score
                scores.update({item : new_score})
        return scores
    
    def item_pos_weight(self, pos_candidate, pos_item, lambda_inh):
        diff = abs(pos_candidate - pos_item)
        return exp(-diff / lambda_inh)
    
    def clear(self):
        self.data = None
        self.session_item_map  = dict()
        self.session_last_time = dict()
        self.item_session_map  = dict()

    def support_users(self):
        '''
        whether it is a session-based or session-aware algorithm
        (if returns True, method "predict_with_training_data" must be defined as well)
        Parameters
        --------
        Returns
        --------
        True : if it is session-aware
        False : if it is session-based
        '''
        return False