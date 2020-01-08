_hyponym(Y,X) :- _hypernym(X,Y)
_hypernym(Y,X) :- _hyponym(X,Y)
