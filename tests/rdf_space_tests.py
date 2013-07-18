from nose.tools import *
import numpy as np
from numpy.linalg import *
from numpy.testing import *
import rdfspace
from rdfspace.space import Space

def test_init():
    rdf_space = Space('tests/example.n3', ignore_inverse=False)
    assert_equal(rdf_space._path_to_rdf, 'file:tests/example.n3')
    assert_equal(rdf_space._format, 'ntriples')
    assert_equal(rdf_space._ignored_predicates, ['http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/homepage'])
    assert_equal(rdf_space._predicates, None)
    assert_equal(rdf_space.uri_index, {
        'http://dbpedia.org/resource/Category:Star_Trek': 0,
        'http://dbpedia.org/resource/Category:Categories_named_after_television_series': 1,
        'http://dbpedia.org/resource/Category:Futurama': 2,
        'http://dbpedia.org/resource/Category:New_York_City_in_fiction': 3,
        'http://dbpedia.org/resource/Category:Comic_science_fiction': 4,
    })
    assert_equal(rdf_space._adjacency_value, 1.0)
    assert_equal(rdf_space._diagonal_value, 10.0)
    assert_equal(rdf_space._adjacency.shape, (5, 5))
    assert_equal(rdf_space._adjacency[0,0], 10/np.sqrt(10 ** 2 + 1))
    assert_equal(rdf_space._adjacency[0,1], 1/np.sqrt(10 ** 2 + 2))
    assert_equal(rdf_space._adjacency[0,2], 0)
    assert_equal(rdf_space._adjacency[0,3], 0)
    assert_equal(rdf_space._adjacency[0,4], 0)
    assert_equal(rdf_space._adjacency[1,0], 1/np.sqrt(10 ** 2 + 1))
    assert_equal(rdf_space._adjacency[1,1], 10/np.sqrt(10 ** 2 + 2))
    assert_equal(rdf_space._adjacency[1,2], 1/np.sqrt(10 ** 2 + 3))
    assert_equal(rdf_space._adjacency[1,3], 0)
    assert_equal(rdf_space._adjacency[1,4], 0)
    assert_equal(rdf_space._adjacency[2,0], 0)
    assert_equal(rdf_space._adjacency[2,1], 1/np.sqrt(10 ** 2 + 2))
    assert_equal(rdf_space._adjacency[2,2], 10/np.sqrt(10 ** 2 + 3))
    assert_equal(rdf_space._adjacency[2,3], 1/np.sqrt(10 ** 2 + 1))
    assert_equal(rdf_space._adjacency[2,4], 1/np.sqrt(10 ** 2 + 1))
    assert_equal(rdf_space._adjacency[3,0], 0)
    assert_equal(rdf_space._adjacency[3,1], 0)
    assert_equal(rdf_space._adjacency[3,2], 1/np.sqrt(10 ** 2 + 3))
    assert_equal(rdf_space._adjacency[3,3], 10/np.sqrt(10 ** 2 + 1))
    assert_equal(rdf_space._adjacency[3,4], 0)
    assert_equal(rdf_space._adjacency[4,0], 0)
    assert_equal(rdf_space._adjacency[4,1], 0)
    assert_equal(rdf_space._adjacency[4,2], 1/np.sqrt(10 ** 2 + 3))
    assert_equal(rdf_space._adjacency[4,3], 0)
    assert_equal(rdf_space._adjacency[4,4], 10/np.sqrt(10 ** 2 + 1))

def test_index():
    rdf_space = Space('tests/example.n3')
    assert_equal(rdf_space.index('http://dbpedia.org/resource/Category:Futurama'), 2)

def test_to_vector():
    rdf_space = Space('tests/example.n3', rank=2)
    assert_equal(rdf_space.to_vector('http://dbpedia.org/resource/Category:Futurama').shape, (2,))

def test_similarity():
    rdf_space = Space('tests/example.n3')
    # Overriding _ut
    rdf_space._ut = np.array([[0,1,0,0],[1,0,0,0],[0,1,0,0],[1,1,1,1]], dtype=float).T
    # Overriding uri_index
    rdf_space.uri_index = {'http://0': 0, 'http://1': 1, 'http://2': 2, 'http://3': 3}

    assert_equal(rdf_space.similarity('http://0', 'http://0'), 1.0)
    assert_equal(rdf_space.similarity('http://0', 'http://1'), 0)
    assert_equal(rdf_space.similarity('http://0', 'http://2'), 1.0)
    assert_equal(rdf_space.similarity('http://0', 'http://3'), 0.5)

def test_centroid():
    rdf_space = Space('tests/example.n3')
    # Overriding _ut
    rdf_space._ut = np.array([[0,1,0,0],[1,0,0,0],[0,1,0,0],[1,1,1,1]], dtype=float).T
    # Overriding uri_index
    rdf_space.uri_index = {'http://0': 0, 'http://1': 1, 'http://2': 2, 'http://3': 3}

    centroid = rdf_space.centroid(['http://0', 'http://1', 'http://2', 'http://3'])
    assert_array_equal(centroid, np.array([0.5, 0.75, 0.25, 0.25]))
    centroid = rdf_space.centroid(['http://0', 'http://3'])
    assert_array_equal(centroid, np.array([0.5, 1, 0.5, 0.5]))
    centroid = rdf_space.centroid(['http://0', 'http://1'])
    assert_array_equal(centroid, np.array([0.5, 0.5, 0, 0]))
    centroid = rdf_space.centroid(['http://0', 'http://6'])
    assert_array_equal(centroid, np.array([0, 1, 0, 0]))
    centroid = rdf_space.centroid([])
    assert_array_equal(centroid, None)

def test_save_and_load():
    rdf_space = Space('tests/example.n3')
    rdf_space._ut = np.random.rand(5, 5)
    rdf_space._ut_shape = (5, 5)
    rdf_space._s = np.random.rand(5, 5)
    rdf_space._s_shape = (5, 5)
    rdf_space._vt = np.random.rand(5, 5)
    rdf_space._vt_shape = (5, 5)
    adj = rdf_space._adjacency
    rdf_space.save('tests/example-space')
    assert_array_equal(rdf_space._adjacency, adj)

    space = Space.load('tests/example-space')
    assert_equal(space.uri_index, rdf_space.uri_index)
    assert_equal(space._ut[2,3], rdf_space._ut[2,3])
    assert_equal(space._s[2,2], rdf_space._s[2,2])
    assert_equal(space._vt[2,3], rdf_space._vt[2,3])
    assert_equal(space._adjacency, None)

    try:
        space = Space.load('foo')
        assert_true(false)
    except Exception as e:
        assert_equal(e.args[0], 'No such directory')
    
