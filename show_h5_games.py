from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
from gooop import Goban

# Initialize the Goban.
go_board = Goban(13)

# Build the empty board.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('xkcd:sandy')
ax.axis('equal')
ax.set(xlim=(0, 12), ylim=(0, 12))
ax.set_xticks(np.arange(13))
ax.set_yticks(np.arange(13))
ax.grid(color="xkcd:black", linewidth=1)
ax.set_axisbelow(True)
ax.set_aspect('equal', 'box')


def reset_plot():
    ax.clear()
    ax.set_facecolor('xkcd:sandy')
    ax.axis('equal')
    ax.set(xlim=(0, 12), ylim=(0, 12))
    ax.set_xticks(np.arange(13))
    ax.set_yticks(np.arange(13))
    ax.grid(color="xkcd:black", linewidth=1)
    ax.set_axisbelow(True)
    ax.set_aspect('equal', 'box')


def set_stone_of_color(row, col, stone_color):
    """
    Draw a stone of a particular color on the board at the given spot.
    """
    if stone_color == "B":
        stone = patches.Circle([col, row], radius=0.5, facecolor="xkcd:black") 
    else:
        stone = patches.Circle([col, row], radius=0.5, facecolor="xkcd:white")
    ax.add_patch(stone)


def attempt_move(row, col):
    """
    Try making a move and get the updated board state. Reset all stones on the display.
    """
    reset_plot()
    adjusted_row = 12 - row
    go_board.make_move(adjusted_row, col)
    current_board = go_board.get_current_board()
    print(current_board)
    for s_row in range(13):
        for s_col in range(13):
            stone_val = current_board[s_row, s_col]
            if stone_val == go_board.white:
                print("Showing white stone")
                set_stone_of_color(12 - s_row, s_col, "W")
            elif stone_val == go_board.black:
                set_stone_of_color(12 - s_row, s_col, "B")
                print("Showing black stone")


def onclick(event):
    """
    If the user clicks on an intersection (within 20 %)
    add a stone to that spot if its a legal move.
    """
    
    threshold_low = 0.4
    threshold_high = 1.0 - threshold_low

    x_pos = int(event.xdata)
    x_extra = event.xdata - x_pos
    y_pos = int(event.ydata)
    y_extra = event.ydata - y_pos

    if x_extra < threshold_low and y_extra < threshold_low:
        attempt_move(y_pos, x_pos)
    elif x_extra < threshold_low and y_extra > threshold_high:
        attempt_move(y_pos + 1, x_pos)
    elif x_extra > threshold_high and y_extra < threshold_low:
        attempt_move(y_pos, x_pos + 1)
    elif x_extra > threshold_high and y_extra > threshold_high:
        attempt_move(y_pos + 1, x_pos + 1)

    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


# Display the plot. Start looping.
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
