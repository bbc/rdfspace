# rdfspace
# 
# Copyright (c) 2013 British Broadcasting Corporation
# 
# Licensed under the GNU Affero General Public License version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.gnu.org/licenses/agpl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.linalg as linalg
from scipy import *
from scipy.sparse import *
from sparsesvd import sparsesvd
from numpy.linalg import *
from operator import itemgetter
import RDF
import cPickle as pickle
import os

class Space(object):

    def __init__(self, path_to_rdf, format='ntriples', ignored_predicates=['http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/homepage'], predicates=None, rank=50, ignore_inverse=False, adjacency_value=1.0, diagonal_value=10.0, normalisation='norm'):
        self._path_to_rdf = 'file:' + path_to_rdf
        self._format = format
        self._ignored_predicates = ignored_predicates
        self._predicates = predicates
        self._rank = rank
        self._adjacency_value = adjacency_value
        self._diagonal_value = diagonal_value
        self._ignore_inverse = ignore_inverse
        self._normalisation = normalisation
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
        ij_exists = {}

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
                    ij.append([i, i])
                    data.append(self._diagonal_value)
                    ij_exists[self.ij_key(i,i)] = True
                    norms[i] = [k]
                    k += 1
                    i += 1
                if not uri_index.has_key(o):
                    uri_index[o] = i
                    ij.append([i, i])
                    data.append(self._diagonal_value)
                    ij_exists[self.ij_key(i,i)] = True
                    norms[i] = [k]
                    k += 1
                    i += 1
                m, n = uri_index[s], uri_index[o]
                if not ij_exists.has_key(self.ij_key(m,n)):
                    ij.append([m, n])
                    data.append(self._adjacency_value)
                    ij_exists[self.ij_key(m,n)] = True
                    norms[n].append(k)
                    k += 1
                if not self._ignore_inverse:
                    if not ij_exists.has_key(self.ij_key(n,m)):
                        ij.append([n, m])
                        data.append(self._adjacency_value)
                        ij_exists[self.ij_key(n,m)] = True
                        norms[m].append(k)
                        k += 1
            z += 1
            if z % 100000 == 0:
                print "Processed " + str(z) + " triples..."

        if self._normalisation in ['norm', 'logentropy']:
            print "Normalising..."
            for m in norms.keys():
                values = [ self._adjacency_value for x in range(0, len(norms[m]) - 1) ]
                values.append(self._diagonal_value) # One diagonal value per column, the rest are adjacencies
                p = 1.0 / norm(values)
                if self._normalisation == 'norm':
                    for n in norms[m]:
                        data[n] = data[n] * p
                elif self._normalisation == 'logentropy':
                    p = (1 + p * log(p) * i / log(i))
                    for n in norms[m]:
                        data[n] = data[n] * log(data[n] + 1) * p
        else:
            print "Skipping normalisation..."

        data = array(data)
        ij = array(ij)
        ij = ij.T
        self.uri_index = uri_index
        self._adjacency = csc_matrix((data, ij), shape=(i,i))

    def ij_key(self, i, j):
        return str(i) + '!' + str(j)

    def projections(self):
        if self._ut is None:
            self._ut, self._s, self._vt = sparsesvd(self._adjacency, self._rank) 
            (self._ut_shape, self._s_shape, self._vt_shape) = (self._ut.shape, self._s.shape, self._vt.shape)
        return self._ut.T

    def index(self, uri):
        return self.uri_index[uri]

    def cosine(self, v1, v2):
        if norm(v1) == 0 or norm(v2) == 0:
            return 0
        return dot(v1, v2.T) / (norm(v1) * norm(v2))

    def similarity_ij(self, i, j):
        projections = self.projections()
        return self.cosine(projections[i], projections[j])

    def similarity(self, uri_1, uri_2):
        return self.similarity_ij(self.uri_index[uri_1], self.uri_index[uri_2])

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

    def to_vector(self, uri):
        return self.projections()[self.uri_index[uri]]

    def centrality(self, uri):
        return self.to_vector(uri)[0]

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
