p(X, Y) :- q(X, Y) <50>
p(X, Y) :- q(X, Y), r(X, Y) <50>
p(X, Y) :- q(X, Y), r(Y, X) <50>
p(X, Y) :- q(Y, X) <50>
p(X, Y) :- q(Y, X), r(Y, X) <50>
p(X, Y) :- q(Y, X), r(X, Y) <50>
p(X, Y) :- q(X, Z), r(Z, Y) <50>
p(X, Y) :- q(Y, Z), r(Z, X) <50>
