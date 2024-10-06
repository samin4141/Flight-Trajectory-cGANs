import random
import math


def delta_maker(lines):
    delta_lines = []
    for line in lines:
        new_line = []
        for line_dim in line:
            delta = []
            for i, item in enumerate(line_dim):
                if i == 0:
                    prev = item
                    continue
                delta.append(item-prev)
                prev = item
            new_line.append(delta)
        delta_lines.append(new_line)
    return delta_lines

def rev_delta_cart_maker(lines):
    newlines = []
    for line in lines:
        new_line = []
        for line_dim in line:
            delta = [0]
            prev = 0
            for i, item in enumerate(line_dim):
                delta.append(item+prev)
                prev = item+prev
            new_line.append(delta)
        newlines.append(new_line)
    return newlines



def delta_cyli_maker(lines):
    delta_lines = []
    for i, line in enumerate(lines):
        r = []
        theta = []
        prev_r = 0
        prev_theta = 0
        theta_offset = 0
        for j in range(1, len(line[0])):
            r.append((line[0][j]**2 + line[1][j]**2)**0.5 - prev_r)
            prev_r = (line[0][j]**2 + line[1][j]**2)**0.5
            
            #We need to place checks to avoid jumps in theta...
            if math.atan2(line[1][j], line[0][j]) - prev_theta >= 2*3.14: #If the offset is greater than pi (-0.05pi -> 2pi)
                min_jump_amm = math.atan2(line[1][j], line[0][j]) - prev_theta
                for i in range(-2, 3):
                    if math.atan2(line[1][j], line[0][j]) - prev_theta + 2*3.14*i < min_jump_amm:
                        min_jump_amm = math.atan2(line[1][j], line[0][j]) - prev_theta + 2*3.14*i
                        theta_offset = 3.14*i
                    
            theta.append(math.atan2(line[1][j], line[0][j]) - prev_theta + 2*3.14*i)
            prev_theta = math.atan2(line[1][j], line[0][j])

        delta_lines.append([r, theta])
    return delta_lines


def rev_delta_cyli_maker(lines):
    newlines = []
    for i, line in enumerate(lines):
        x = [0]
        y = [0]
        prev_r = 0
        prev_theta = 0
        for j in range(0, len(line[0])):
            new_r = prev_r + line[0][j]
            new_theta = prev_theta + line[1][j]
            
            x.append(new_r * math.cos(new_theta))
            y.append(new_r * math.sin(new_theta))
            prev_r = new_r
            prev_theta = new_theta

        newlines.append([x, y])
    return newlines


## Drawing spirals ###
def draw_curved_line(class_num, theta, length):
    if class_num == 1:
        r = [150+i*random.random()*100/length for i in range(0, length)] #Big spiral
    else:
        r = [150+i*random.random()*100/length for i in range(0, length)] #Small spiral
    return [[0] + [(i*r[i])*math.cos(theta[i])/length+r[i]*(random.random()-0.5)*0.01 for i in range(1, length)], \
          [0] + [(i*r[i])*math.sin(theta[i])/length+r[i]*(random.random()-0.5)*0.01 for i in range(1, length)]]



## Drawing Spiral lines ###
def draw_spiral_line(class_num, theta, length):
    distance = [(i*1000+200*random.random())/length for i in range(0, length)]
    fixed_theta = random.random()*3.14/2
    r = [500+i*random.random()*10/length for i in range(0, length)]
    
    return [[0] + [(distance[i]*math.cos(fixed_theta)) + (i*r[i])*math.cos(theta[i])/length+r[i]*(random.random()-0.5)*0.01 for i in range(1, length)], \
          [0] + [(distance[i]*math.sin(fixed_theta)) + (i*r[i])*math.sin(theta[i])/length+r[i]*(random.random()-0.5)*0.01 for i in range(1, length)], \
          [0] + [distance[i]*0.5+r[i]*0.3 for i in range(1, length)]]


## Drawing straight lines ###
def draw_line(r, theta, length):
  return [[0] + [i*r*math.cos(theta)/length+r*(random.random()-0.5)*0.01 for i in range(1, length)], \
          [0] + [i*r*math.sin(theta)/length+r*(random.random()-0.5)*0.01 for i in range(1, length)]]

