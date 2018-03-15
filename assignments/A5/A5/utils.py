#!/usr/bin/env python

"""
"""

import numpy.random as rnd

n = None
a = None

def rand_in_range(max): # returns integer, max: integer
    return rnd.randint(max)
    
def rand_un(): # returns floating point
    return rnd.uniform()

def rand_norm (mu, sigma): # returns floating point, mu: floating point, sigma: floating point
    return rnd.normal(mu, sigma)
    
def set_n(new_n):
    global n
    n = new_n
def get_n():
    return n

def set_a(new_a):
    global a
    a = new_a
def get_a():
    return a 