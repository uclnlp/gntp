neighborOf(X,Y) :- neighborOf(Y,X)
locatedIn(X,Y) :- locatedIn(X,Z), locatedIn(Z,Y)
