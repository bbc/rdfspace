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
    >>> space.similarity('http://dbpedia.org/resource/JavaScript', 'http://dbpedia.org/resource/ECMAScript')
    >>> space.similarity('http://dbpedia.org/resource/Albert_Camus', 'http://dbpedia.org/resource/JavaScript')

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
