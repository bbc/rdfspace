import numpy as np
import scipy.linalg as linalg
from scipy import *
from scipy.sparse import *
from sparsesvd import sparsesvd
from numpy.linalg import *
from operator import itemgetter
import RDF
import pickle
import os

class Space(object):

    def __init__(self, path_to_rdf, format='ntriples', ignored_predicates=[], predicates=None, rank=50, ignore_inverse=True):
        self._path_to_rdf = 'file:' + path_to_rdf
        self._format = format
        self._ignored_predicates = ignored_predicates
        self._predicates = predicates
        self._rank = rank
        self._ignore_inverse = ignore_inverse
        self._adjacency  = None
        self.uri_index = None
        self._ut = None
        self._s = None
        self._vt = None
        print "Creating sparse adjacency matrix..."
        self.generate_vector_space()
        print "Generating SVD..."
        self.projections()

    def generate_vector_space(self):
        if self._adjacency != None and self.uri_index != None:
            return

        parser = RDF.Parser(name=self._format)
        stream = parser.parse_as_stream(self._path_to_rdf)

        uri_index = {}
        data = []
        ij = []
        norms = {}

        i = 0
        z = 0
        k = 0
        for statement in stream:
            p = str(statement.predicate.uri)
            if statement.object.is_resource() and (not self._predicates or p in self._predicates) and p not in self._ignored_predicates:
                if statement.subject.is_blank():
                    s = str(statement.subject)
                else:
                    s = str(statement.subject.uri)
                if statement.object.is_blank():
                    o = str(statement.object)
                else:
                    o = str(statement.object.uri)
                if not uri_index.has_key(s):
                    uri_index[s] = i
                    i += 1
                if not uri_index.has_key(o):
                    uri_index[o] = i
                    i += 1
                ij.append([uri_index[s], uri_index[o]])
                if not self._ignore_inverse:
                    ij.append([uri_index[o], uri_index[s]])
                if norms.has_key(uri_index[o]):
                    norms[uri_index[o]].append(k)
                else:
                    norms[uri_index[o]] = [k]
                data.append(1.0)
                k += 1
                if not self._ignore_inverse:
                    if norms.has_key(uri_index[s]):
                        norms[uri_index[s]].append(k)
                    else:
                        norms[uri_index[s]] = [k]
                    data.append(1.0)
                    k += 1
            z += 1
            if z % 100000 == 0:
                print "Processed " + str(z) + " triples..."

        print "Normalising..."
        for m in norms.keys():
            p = 1.0 / sqrt(len(norms[m]))
            # should I switch to Log Entropy weighting functions?
            # norm = (1 + p * log(p) * i / log(i))
            for n in norms[m]:
                data[n] = data[n] * p # log(data[n] + 1) * norm

        data = array(data)
        ij = array(ij)
        ij = ij.T
        self.uri_index = uri_index
        self._adjacency = csc_matrix((data, ij), shape=(i,i))

    def projections(self):
        if self._ut == None:
            self._ut, self._s, self._vt = sparsesvd(self._adjacency, self._rank) 
            (self._ut_shape, self._s_shape, self._vt_shape) = (self._ut.shape, self._s.shape, self._vt.shape)
        return self._ut.T

    def index(self, uri):
        return self.uri_index[uri]

    def cosine(self, v1, v2):
        if norm(v1) == 0 or norm(v2) == 0:
            return 0
        return dot(v1, v2.T) / (norm(v1) * norm(v2))

    def distance_ij(self, i, j):
        projections = self.projections()
        return self.cosine(projections[i], projections[j])

    def distance(self, uri_1, uri_2):
        return self.distance_ij(self.uri_index[uri_1], self.uri_index[uri_2])

    def centroid_ij(self, indexes):
        if not indexes:
            return None
        projections = self.projections()
        return np.mean(projections[indexes], 0)

    def centroid(self, uris):
        if not uris:
            return None
        indexes = []
        for uri in uris:
            # We drop URIs we don't know about
            if self.uri_index.has_key(uri):
                indexes.append(self.uri_index[uri])
        return self.centroid_ij(indexes)

    def similar(self, uri, limit=10):
        projected = self.projections()
        similarities = {}
        v = projected[self.uri_index[uri]]
        for key in self.uri_index.keys():
            similarities[key] = self.cosine(v, projected[self.uri_index[key]])
        similarities = sorted(similarities.items(), key=itemgetter(1))
        similarities.reverse()
        return similarities[0:limit]

    def save(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # We memmap big matrices, as pickle eats the whole RAM
        # We don't save the full adjacency matrix
        ut_m = np.memmap(os.path.join(dirname, 'ut.dat'), dtype='float64', mode='w+', shape=self._ut_shape)
        ut_m[:] = self._ut[:]
        s_m = np.memmap(os.path.join(dirname, 's.dat'), dtype='float64', mode='w+', shape=self._s_shape)
        s_m[:] = self._s[:]
        vt_m = np.memmap(os.path.join(dirname, 'vt.dat'), dtype='float64', mode='w+', shape=self._vt_shape)
        vt_m[:] = self._vt[:]
        (adjacency, ut, s, vt) = (self._adjacency, self._ut, self._s, self._vt)
        (self._adjacency, self._ut, self._s, self._vt) = (None, None, None, None)
        f = open(os.path.join(dirname, 'space.dat'), 'w')
        pickle.dump(self, f)
        f.close()
        (self._adjacency, self._ut, self._s, self._vt) = (adjacency, ut, s, vt)

    @staticmethod
    def load(dirname):
        if os.path.exists(dirname):
            f = open(os.path.join(dirname, 'space.dat'))
            space = pickle.load(f)
            f.close()
            space._ut = np.memmap(os.path.join(dirname, 'ut.dat'), dtype='float64', mode='r', shape=space._ut_shape)
            space._s = np.memmap(os.path.join(dirname, 's.dat'), dtype='float64', mode='r', shape=space._s_shape)
            space._vt = np.memmap(os.path.join(dirname, 'vt.dat'), dtype='float64', mode='r', shape=space._vt_shape)
            return space
        else:
            raise Exception('No such directory')
