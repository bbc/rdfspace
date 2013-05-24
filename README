RDFSpace
========

This Python library helps generating a low-dimensional vector space from very large RDF graphs in a reasonable time.
For example it is possible to process large sections of DBpedia on a commodity laptop.

Getting started
---------------

Running the tests:

    $ nosetests

Installing:
 
    $ python setup.py install

Example use
-----------

    $ cd examples
    $ gunzip influencedby.nt.gz
    $ python
    >>> import rdfspace
    >>> from rdfspace.space import Space
    >>> space = Space('influencedby.nt', ignored_predicates=['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'], rank=50)
    >>> space.distance('http://dbpedia.org/resource/JavaScript', 'http://dbpedia.org/resource/ECMAScript')
    >>> space.distance('http://dbpedia.org/resource/Albert_Camus', 'http://dbpedia.org/resource/JavaScript')

Alternatively, a subset of it is available in the examples/ directory.

How it works
------------

RDFSpace construct a sparse adjacency matrix from an input RDF file.
We perform Singular Value Decomposition on this sparse adjacency matrix
to approximate this space, which gives us a lower-dimensional space
in which URIs that are close to each other in the origin RDF graph
will have a high cosine similarity, and URIs that are far from each other in
the origin RDF graph will have a low cosine similarity.


Licensing terms and authorship
------------------------------

See 'COPYING' and 'AUTHORS' files.
