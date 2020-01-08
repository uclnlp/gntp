_hyponym(Y,X) :- _hypernym(X,Y)
_part_of(Y,X) :- _has_part(X,Y)
_hypernym(Y,X) :- _hyponym(X,Y)
_has_part(Y,X) :- _part_of(X,Y)
