# requirement
<!-- * map -> queue(for memory and for visualizing result on-time) -->
* feature vector board -> something
* it takes a lot of time for learning, so it is needed to decrease time.
* in this algorithm,
*  it learns on time(time depending).

# result
learning 1000 num 30
result:
```
[1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
1/ 29/ 0 -- learning VS random
```

learning 30 num 8
```
10 VS 30
[-1, 1, 1, 1, -1, -1, -1, -1]
for init,  3 -- win,  5 -- lose,  0 -- draw
52.23037004470825 seconds

0 VS 30
51.838733434677124 seconds
[-1, 1, -1, -1, -1, -1, -1, -1]
for init,  1 -- win,  7 -- lose,  0 -- draw
```

10 VS 1000
```
black
 abcdefgh
1
2
3  O   O
4  OXO O
5  X   X
6 X
7X
play time 0.22675013542175293
812.236184835434 seconds
[-1]
for init,  0 -- win,  1 -- lose,  0 -- draw
```
