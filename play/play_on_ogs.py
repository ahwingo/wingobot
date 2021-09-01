"""
WORK IN PROGRESS: Based on the GNU Go example.
This script manages the WingoBot as it plays live on OGS.
"""

# Python Standard
import subprocess
import time

# Third Party
from gtp import GoTextPipe, GoTextNetwork


def check(go):
    for _ in range(2):
        print(str(go.genmove('black').decode("utf-8")))
        print(str(go.genmove('white').decode("utf-8")))
        print(str(go.estimate_score().decode("utf-8")))
        print(str(go.showboard().decode("utf-8")))

    print(repr(go.boardsize(5)))
    print(repr(go.clear_board()))
    print(go.showboard())
    print(repr(go.play('black', 'A5')))
    print(repr(go.play('white', 'A4')))
    print(repr(go.estimate_score()))
    print(go.showboard())

    go.close()

cmd = "gnugo --mode gtp --gtp-listen 127.0.0.1:50001 --boardsize 13"
p = subprocess.Popen(cmd.split())
time.sleep(1)

go = GoTextNetwork('localhost', 50001)
check(go)

# Stop the gnugo instance.
p.kill()

