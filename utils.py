import numpy as np
import random
import torch
import os
from shutil import copyfile
from gym import spaces

def stable_normalizer(x,temp):
    x = (x / np.max(x))**temp
    return np.abs(x/np.sum(x))

def check_space(space):    
    if isinstance(space,spaces.Box):
        dim = space.shape
        discrete = False    
    elif isinstance(space,spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError('This type of space is not supported')
    return dim, discrete

def store_safely(folder,name,to_store):
    new_name = folder+name+'.npy'
    old_name = folder+name+'_old.npy'
    if os.path.exists(new_name):
        copyfile(new_name,old_name)
    np.save(new_name,to_store)
    if os.path.exists(old_name):            
        os.remove(old_name)

def get_base_env(env):
    while hasattr(env,'env'):
        env = env.env
    return env

def copy_atari_state(env):
    env = get_base_env(env)
    return env.clone_full_state()

def restore_atari_state(env,snapshot):
    env = get_base_env(env)
    env.restore_full_state(snapshot)

def is_atari_game(env):
    env = get_base_env(env)
    return hasattr(env,'ale')

    
class Database():
    
    def __init__(self,max_size,batch_size):
        self.max_size = max_size        
        self.batch_size = batch_size
        self.clear()
        self.sample_array = None
        self.sample_index = 0
    
    def clear(self):
        self.experience = []
        self.insert_index = 0
        self.size = 0
    
    def store(self,experience):
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size +=1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def store_from_array(self,*args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)
        
    def reshuffle(self):
        self.sample_array = np.arange(self.size)
        random.shuffle(self.sample_array)
        self.sample_index = 0
                            
    def __iter__(self):
        return self

    def __next__(self):
        if (self.sample_index + self.batch_size > self.size) and (not self.sample_index == 0):
            self.reshuffle()
            raise(StopIteration)
          
        if (self.sample_index + 2*self.batch_size > self.size):
            indices = self.sample_array[self.sample_index:]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[self.sample_index:self.sample_index+self.batch_size]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size
        
        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add) 
        return tuple(arrays)
            
    next = __next__
    


def symmetric_remove(x,n):
    odd = is_odd(n)
    half = int(n/2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x

def is_odd(number):
    return bool(number & 1)

def smooth(y,window,mode):   
    return np.convolve(y, np.ones(window)/window, mode=mode)