Total for 100 its (including logging costs)
Async offload (spawing new processes):
    65.20386981964111

Sync offload:
    21.174951314926147

concurrent.futures:
    21.826061964035034 


bsize == 64:
concurrent.futures:
    66.77062773704529
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
6    0.000    0.000    7.583    1.264 _tracker.py:102(offload_global_stash)
6    0.000    0.000    7.583    1.264 _tracker.py:112(launch_background_process)
320270/6    1.379    0.000    7.402    1.234 {built-in method _pickle.dump}

sync:
    63.53866958618164
6    0.000    0.000    3.894    0.649 _tracker.py:102(offload_global_stash)
6    0.026    0.004    0.026    0.004 {built-in method _pickle.dump}



All Stats, bsize 1, log inc = 20 w/ msgpack:
async:
    31.40442156791687
sync:
    32.61379790306091


All Stats, bsize 64, log inc=1 w/ msgpack:
async:
    107.76926040649414
sync:
    109