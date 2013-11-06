import rdfspace
from rdfspace.space import Space

space = Space('influencedby.nt', ignored_predicates=['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'], rank=50)

print "Similarity betwen JavaScript and ECMAScript:"
print space.similarity('http://dbpedia.org/resource/JavaScript', 'http://dbpedia.org/resource/ECMAScript')

print "Eigenvector centrality of JavaScript:"
print space.centrality('http://dbpedia.org/resource/JavaScript')

print "Similarity between Albert Camus and JavaScript:"
print space.similarity('http://dbpedia.org/resource/Albert_Camus', 'http://dbpedia.org/resource/JavaScript')

print "Eigenvector centrality of Albert Camus"
print space.centrality('http://dbpedia.org/resource/Albert_Camus')
