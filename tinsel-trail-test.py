from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('xkcd:black')
ax.axis('equal')
ax.set(xlim=(0, 12*4), ylim=(0, 12*7))
ax.set_axisbelow(True)
ax.set_aspect('equal', 'box')
all_leds = []

def draw_tree():
    leaves = patches.Polygon(np.asarray([[0, 12], [48, 12], [24, 7*12]]), facecolor="xkcd:darkgreen")
    trunk = patches.Polygon(np.asarray([[18, 0], [30, 0], [30, 12], [18, 12]]), facecolor="xkcd:brown")
    ax.add_patch(leaves)
    ax.add_patch(trunk)



def draw_led(x_pos, y_pos):
    led = patches.Circle([x_pos, y_pos], radius=0.25, facecolor="xkcd:pink")
    ax.add_patch(led)
    all_leds.append(led)
    return led


def draw_box(start_pos, led_spacing, num_leds, box_width):
    start_x, start_y, start_z = start_pos
    curr_x, curr_y, curr_z = start_pos
    direction = 1
    for idx in range(num_leds):
        if curr_x > start_x + box_width:
            curr_x = start_x + box_width
            direction *= -1
            curr_y += led_spacing
        elif curr_x < start_x:
            direction *= -1
            curr_x = start_x
            curr_y += led_spacing
        led = draw_led(curr_x, curr_y)
        curr_x += direction * led_spacing


def build_array(start_x, start_y, led_spacing, num_rows, num_cols):
    direction = 1
    top = num_cols*led_spacing + start_y
    left = start_x
    for row in range(num_rows):
        for col in range(num_cols):





draw_tree()
draw_box([12, 18, 0], 1.5, 17**2, 24)

plt.show()
