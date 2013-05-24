import rdfspace
from rdfspace.space import Space

space = Space('influencedby.nt', ignored_predicates=['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'], rank=50)

print "Distance betwen JavaScript and ECMAScript:"
print space.distance('http://dbpedia.org/resource/JavaScript', 'http://dbpedia.org/resource/ECMAScript')

print "Distance between Albert Camus and JavaScript:"
print space.distance('http://dbpedia.org/resource/Albert_Camus', 'http://dbpedia.org/resource/JavaScript')
