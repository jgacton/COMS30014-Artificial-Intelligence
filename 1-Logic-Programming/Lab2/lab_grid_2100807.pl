% True if A is a possible movement direction
m(A) :-
  A == n ; A == e ; A == s ; A == w.

% True if p(X,Y) is on the board
on_board(p(X,Y)) :-
    ailp_grid_size(S) , X > 0 , Y > 0 , X =< S , Y =< S.

% True if p(X1,Y1) is one step in direction M from p(X,Y) (no bounds check)
pos_step(p(X,Y), M, p(X1,Y1)) :-
    m(M) , (
    (M == n , Y1 is Y+1) ; (M == e , Y1 is Y+1) ;
    (M == s , X1 is X+1) ; (M == w , Y1 is Y-1) ).

% True if NPos is one step in direction M from Pos (with bounds check)
new_pos(Pos,M,NPos) :-
    pos_step(Pos,M,NPos) , on_board(Pos) , on_board(NPos).

% True if a L has the same length as the number of squares on the board
complete(L) :-
    ailp_grid_size(S), length(L, N), N is S*S.

% Perform a sequence of moves creating a spiral pattern, return the moves as L
spiral(L) :-
    (L = [] , my_agent(A) , get_agent_position(A,P) , spiral([P])) ;
    (L = [F|K] , K = [] , F = p(X,Y) , (
    (new_pos(p(X,Y) , n , p(X,Y+1)) , append([p(X,Y+1)],L,R) , spiral(R)) ;
    (new_pos(p(X,Y) , e , p(X+1,Y)) , append([p(X+1,Y)],L,R) , spiral(R)) ;
    (new_pos(p(X,Y) , s , p(X,Y-1)) , append([p(X,Y-1)],L,R) , spiral(R)) ;
    (new_pos(p(X,Y) , w , p(X-1,Y)) , append([p(X-1,Y)],L,R) , spiral(R)))) ;
    (L = [H|T], T = [H2|T2], H = p(X2,Y2), H2 = p(X3,Y3)) ;
    complete(L).

% first call of spiral(L) is with start position e.g. spiral([p(1,1)]).
% spiral then figures out next move, appends to head of list, then recursively
% calls spiral([p(2,1),p(1,1)]), keeps recursively calling and adding moves
% to head of list until no move can be made, at which point return true.
