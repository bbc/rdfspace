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
import re
import dbm

class Space(object):

    def __init__(self, path_to_rdf, index_dir = None, format='ntriples', ignored_predicates=['http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/homepage'], predicates=None, rank=50, ignore_inverse=False, adjacency_value=1.0, diagonal_value=10.0, normalisation='norm', prefix = 'http://dbpedia.org/resource/'):
        self._path_to_rdf = 'file:' + path_to_rdf
        self._index_dir = index_dir
        self._format = format
        self._ignored_predicates = ignored_predicates
        self._predicates = predicates
        self._rank = rank
        self._adjacency_value = adjacency_value
        self._diagonal_value = diagonal_value
        self._ignore_inverse = ignore_inverse
        self._normalisation = normalisation
        self._prefix = prefix
        self._escaped_prefix = re.escape(prefix)
        self._adjacency  = None
        self._uri_index = None
        self._index_uri = None
        self._ut = None
        self._s = None
        self._vt = None
        print "Creating sparse adjacency matrix..."
        self.generate_vector_space()
        print "Generating SVD..."
        self.projections()

    def generate_vector_space(self):
        """Generate a vector space from an RDF file"""
        if self._adjacency != None and self._uri_index != None and self._index_uri != None:
            return

        parser = RDF.Parser(name=self._format)
        stream = parser.parse_as_stream(self._path_to_rdf)

        if self._index_dir is None:
            uri_index = {}
            index_uri = {}
        else:
            if not os.path.exists(self._index_dir):
                os.makedirs(self._index_dir)
            uri_index = dbm.open(os.path.join(self._index_dir, 'uri_index'), 'n')
            index_uri = dbm.open(os.path.join(self._index_dir, 'index_uri'), 'n')

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
                    if self._escaped_prefix:
                        s = re.sub('^' + self._escaped_prefix, '^', s)
                if statement.object.is_blank():
                    o = str(statement.object)
                else:
                    o = str(statement.object.uri)
                    if self._escaped_prefix:
                        o = re.sub('^' + self._escaped_prefix, '^', o)
                if not uri_index.has_key(s):
                    uri_index[s] = str(i)
                    index_uri[str(i)] = s
                    ij.append([i, i])
                    data.append(self._diagonal_value)
                    ij_exists[self.ij_key(i,i)] = True
                    norms[i] = [k]
                    k += 1
                    i += 1
                if not uri_index.has_key(o):
                    uri_index[o] = str(i)
                    index_uri[str(i)] = o
                    ij.append([i, i])
                    data.append(self._diagonal_value)
                    ij_exists[self.ij_key(i,i)] = True
                    norms[i] = [k]
                    k += 1
                    i += 1
                m, n = int(uri_index[s]), int(uri_index[o])
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
        self._uri_index = uri_index
        self._index_uri = index_uri
        self._adjacency = csc_matrix((data, ij), shape=(i,i))

    def ij_key(self, i, j):
        return str(i) + '!' + str(j)

    def projections(self):
        """Get the set of vectors for all URIs"""
        if self._ut is None:
            self._ut, self._s, self._vt = sparsesvd(self._adjacency, self._rank) 
            (self._ut_shape, self._s_shape, self._vt_shape) = (self._ut.shape, self._s.shape, self._vt.shape)
        return self._ut.T

    def index(self, uri):
        """Index of an URI"""
        if self._escaped_prefix:
            uri = re.sub('^' + self._escaped_prefix, '^', uri)
        if self._uri_index.has_key(uri):
            return int(self._uri_index[uri])
        else:
            return None

    def uri(self, index):
        """Uri corresponding to an index"""
        index = str(index)
        if self._index_uri.has_key(index):
            uri = self._index_uri[index]
            if self._prefix:
                uri = re.sub('^\^', self._prefix, uri)
            return uri
        else:
            return None

    def cosine(self, v1, v2):
        """Cosine similarity between two vectors"""
        if norm(v1) == 0 or norm(v2) == 0:
            return 0
        return dot(v1, v2.T) / (norm(v1) * norm(v2))

    def similarity_ij(self, i, j):
        """Cosine similarity between two indexes"""
        projections = self.projections()
        return self.cosine(projections[i], projections[j])

    def similarity(self, uri_1, uri_2):
        """Cosine similarity between two URIs"""
        return self.similarity_ij(self.index(uri_1), self.index(uri_2))

    def centroid_ij(self, indexes):
        """Get the centroid of a set of indexes"""
        if not indexes:
            return None
        projections = self.projections()
        return np.mean(projections[indexes], 0)

    def centroid(self, uris):
        """Get the centroid of a set of URIs"""
        if not uris:
            return None
        indexes = []
        for uri in uris:
            # We drop URIs we don't know about
            index = self.index(uri)
            if index is not None:
                indexes.append(index)
        return self.centroid_ij(indexes)

    def to_vector(self, uri):
        """Get the vector associated with the given URI"""
        return self.projections()[self.index(uri)]

    def centrality(self, uri):
        """Eigenvector centrality of the given URI"""
        return self.to_vector(uri)[0]

    def similar(self, uri, limit=10):
        """Most similar URIs to a given URI"""
        similarities = {}
        v = self.to_vector(uri)
        dot_products = dot(self.projections(), v)
        norms = np.sum(np.abs(self.projections())**2,axis=-1)**(1./2) * norm(v)
        similarities = dot_products / norms
        return [ (self.uri(index), similarities[index]) for index in similarities.argsort()[-limit:][::-1] ]

    def save(self, dirname = None):
        """Save the current rdfspace to a directory (by default the directory in which indexes are stored)"""
        if dirname is None and self._index_dir is not None:
            dirname = self._index_dir
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
        if self._index_dir is None:
            # The index is in memory, we'll pickle it with the rest
            (adjacency, ut, s, vt) = (self._adjacency, self._ut, self._s, self._vt)
            (self._adjacency, self._ut, self._s, self._vt) = (None, None, None, None)
            f = open(os.path.join(dirname, 'space.dat'), 'w')
            pickle.dump(self, f)
            f.close()
            (self._adjacency, self._ut, self._s, self._vt) = (adjacency, ut, s, vt)
        else:
            # Flushing indexes
            self._uri_index.close()
            self._index_uri.close()
            # The index is stored in dbm, we will exclude it from the pickle
            (adjacency, ut, s, vt) = (self._adjacency, self._ut, self._s, self._vt)
            (self._adjacency, self._ut, self._s, self._vt, self._uri_index, self._index_uri) = (None, None, None, None, None, None)
            f = open(os.path.join(dirname, 'space.dat'), 'w')
            pickle.dump(self, f)
            f.close()
            (self._adjacency, self._ut, self._s, self._vt) = (adjacency, ut, s, vt)
            self._uri_index = dbm.open(os.path.join(dirname, 'uri_index'), 'r')
            self._index_uri = dbm.open(os.path.join(dirname, 'index_uri'), 'r')

    @staticmethod
    def load(dirname):
        """Load an rdfspace instance from a directory"""
        if os.path.exists(dirname):
            f = open(os.path.join(dirname, 'space.dat'))
            space = pickle.load(f)
            f.close()
            space._ut = np.memmap(os.path.join(dirname, 'ut.dat'), dtype='float64', mode='r', shape=space._ut_shape)
            space._s = np.memmap(os.path.join(dirname, 's.dat'), dtype='float64', mode='r', shape=space._s_shape)
            space._vt = np.memmap(os.path.join(dirname, 'vt.dat'), dtype='float64', mode='r', shape=space._vt_shape)
            if os.path.exists(os.path.join(dirname, 'uri_index.db')) and os.path.exists(os.path.join(dirname, 'index_uri.db')):
                # If these files exist the index are stored through dbm
                # If not the indexes will be coming from the pickle
                space._uri_index = dbm.open(os.path.join(dirname, 'uri_index'), 'r')
                space._index_uri = dbm.open(os.path.join(dirname, 'index_uri'), 'r')
            return space
        else:
            raise Exception('No such directory')
