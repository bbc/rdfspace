RDFSpace
========

This Python library helps generating a low-dimensional vector space from very large RDF graphs in a reasonable time.
For example it is possible to process large sections of DBpedia on a commodity laptop. Once this
space is generated, it can be used to compute fast similarities between URIs, or to compute
the Eigenvector Centrality (~pagerank) of URIs.

Getting started
---------------

Setting up:

    $ apt-get install python-pip python-librdf python-numpy python-scipy python-nose

Installing:
 
    $ python setup.py install
    # (You might have to run that twice to get around https://github.com/piskvorky/sparsesvd/pull/4)

Running the tests:

    $ nosetests

Or from pypi:

    $ pip install rdfspace

Example use
-----------

    $ cd examples
    $ gunzip influencedby.nt.gz
    $ python
    >>> import rdfspace
    >>> from rdfspace.space import Space
    >>> space = Space('influencedby.nt', ignored_predicates=['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'], rank=50)
    >>> space.similarity('http://dbpedia.org/resource/JavaScript', 'http://dbpedia.org/resource/ECMAScript')
    >>> space.centrality('http://dbpedia.org/resource/JavaScript')
    >>> space.similarity('http://dbpedia.org/resource/Albert_Camus', 'http://dbpedia.org/resource/JavaScript')
    >>> space.centrality('http://dbpedia.org/resource/Albert_Camus')
    >>> space.similar('http://dbpedia.org/resource/Albert_Camus')

How it works
------------

RDFSpace construct a sparse adjacency matrix from an input RDF file.
We perform Singular Value Decomposition on this sparse adjacency matrix
to approximate this space, which gives us a lower-dimensional space
capturing URI similarities. This space can then be used for a wide range
of uses, e.g. automated tagging, disambiguation, etc.


Licensing terms and authorship
------------------------------

See 'COPYING' and 'AUTHORS' files.

All code here, except where otherwise indicated, is licensed under
the GNU Affero General Public License version 3. This license includes
many restrictions. If this causes a problem, please contact us.
See "AUTHORS" for contact details.
