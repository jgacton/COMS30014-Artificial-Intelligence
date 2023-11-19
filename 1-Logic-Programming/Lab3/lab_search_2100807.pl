% Query to find adjacent sells to position p (using ; to get multiple solutions)
% map_adjacent(p(5,5),X,Y).

% Query to find all adjacent sells to p as a list:
% findall(X,map_adjacent(p(8,5),X,Y),Result).

% Query to find only empty cells around p:
% findall(X,(map_adjacent(p(3,3),X,Y),Y=empty),Result).

% Query to find only empty cells next to my agent:
% findall(X,(my_agent(A),get_agent_position(A,P),map_adjacent(P,X,Y),Y=empty),Result).

% Query to find the OID of the Oracle:
% map_adjacent(p(9,1),p(10,1),Y). args are any cell adj to oracle, oracle pos

% Test if the objective has been completed at a given position
complete(P) :-
    map_adjacent(P,_,Y) , Y=o(1).

bfs([Current|Rest],Goal) :-
    (Current = [H|T] , complete(H)) ;
    (Current = [H|T] , findall(X,(map_adjacent(H,X,Y),Y=empty),Children) ,
    Children = [CH|CT] , append(CH,Current,Next) ,
    append(Next,[Current|Rest],Nextagenda) , bfs(Nextagenda,Next)
    ).

% Perform a BFS to find the nearest oracle
search_bf :-
    my_agent(A),
    get_agent_position(A,P),
    (complete(P) -> true
    ;otherwise   -> bfs([[P]],L) , list_reverse(L, RL), RL = [H|T],
    agent_do_moves(A,T)).
