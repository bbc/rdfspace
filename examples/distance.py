import rdfspace
from rdfspace.space import Space

space = Space('skos_categories_1000.nt', ignored_predicates=['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'])

print "Distance betwen Dwarves and Elves:"
print space.distance('http://dbpedia.org/resource/Category:Middle-earth_Dwarves', 'http://dbpedia.org/resource/Category:Middle-earth_Elves')

print "Distance between Dwarves and Futurama:"
print space.distance('http://dbpedia.org/resource/Category:Middle-earth_Dwarves', 'http://dbpedia.org/resource/Category:Futurama')
