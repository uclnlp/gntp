p(X, Y) :- q(X, Y)  <20>
p(X, Y) :- q(Y, X)  <20>
p(X, Y) :- q(X, Z),r(Z, Y)  <20>
p(X, Y) :- q(X, Y),r(X, Y)  <20>
p(X, Y) :- q(Y, X),r(X, Y)  <20>
