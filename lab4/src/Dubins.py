import math
import numpy as np
import matplotlib.pyplot as plt

show_animation = True

def mod2pi(theta):
  return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)


def pi_2_pi(angle):
  while(angle >= math.pi):
    angle = angle - 2.0 * math.pi

  while(angle <= -math.pi):
    angle = angle + 2.0 * math.pi

  return angle

def LSL(alpha, beta, d):
  sa = math.sin(alpha)
  sb = math.sin(beta)
  ca = math.cos(alpha)
  cb = math.cos(beta)
  c_ab = math.cos(alpha - beta)

  tmp0 = d + sa - sb

  mode = ["L", "S", "L"]
  p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
  if p_squared < 0:
    return None, None, None, mode
  tmp1 = math.atan2((cb - ca), tmp0)
  t = mod2pi(-alpha + tmp1)
  p = math.sqrt(p_squared)
  q = mod2pi(beta - tmp1)

  return t, p, q, mode

def RSR(alpha, beta, d):
  sa = math.sin(alpha)
  sb = math.sin(beta)
  ca = math.cos(alpha)
  cb = math.cos(beta)
  c_ab = math.cos(alpha - beta)

  tmp0 = d - sa + sb
  mode = ["R", "S", "R"]
  p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
  if p_squared < 0:
    return None, None, None, mode
  tmp1 = math.atan2((ca - cb), tmp0)
  t = mod2pi(alpha - tmp1)
  p = math.sqrt(p_squared)
  q = mod2pi(-beta + tmp1)

  return t, p, q, mode

def LSR(alpha, beta, d):
  sa = math.sin(alpha)
  sb = math.sin(beta)
  ca = math.cos(alpha)
  cb = math.cos(beta)
  c_ab = math.cos(alpha - beta)

  p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
  mode = ["L", "S", "R"]
  if p_squared < 0:
    return None, None, None, mode
  p = math.sqrt(p_squared)
  tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
  t = mod2pi(-alpha + tmp2)
  q = mod2pi(-mod2pi(beta) + tmp2)

  return t, p, q, mode

def RSL(alpha, beta, d):
  sa = math.sin(alpha)
  sb = math.sin(beta)
  ca = math.cos(alpha)
  cb = math.cos(beta)
  c_ab = math.cos(alpha - beta)

  p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
  mode = ["R", "S", "L"]
  if p_squared < 0:
    return None, None, None, mode
  p = math.sqrt(p_squared)
  tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
  t = mod2pi(alpha - tmp2)
  q = mod2pi(beta - tmp2)

  return t, p, q, mode

def RLR(alpha, beta, d):
  sa = math.sin(alpha)
  sb = math.sin(beta)
  ca = math.cos(alpha)
  cb = math.cos(beta)
  c_ab = math.cos(alpha - beta)

  mode = ["R", "L", "R"]
  tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
  if abs(tmp_rlr) > 1.0:
    return None, None, None, mode

  p = mod2pi(2 * math.pi - math.acos(tmp_rlr))
  t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
  q = mod2pi(alpha - beta - t + mod2pi(p))
  return t, p, q, mode

# alpha: The opposite of the angle of the end position
# beta: The difference between desired end yaw and the angle of the end position
# d: The straight-line-distance/turning-radius   
def LRL(alpha, beta, d):
  sa = math.sin(alpha)
  sb = math.sin(beta)
  ca = math.cos(alpha)
  cb = math.cos(beta)
  c_ab = math.cos(alpha - beta)

  mode = ["L", "R", "L"]
  tmp_lrl = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (- sa + sb)) / 8.0
  if abs(tmp_lrl) > 1:
    return None, None, None, mode
  p = mod2pi(2 * math.pi - math.acos(tmp_lrl))
  t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
  q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

  return t, p, q, mode

# Don't call this directly, use dubins_path_planning
# ex: The end x position
# ey: The end y position
# eyaw: The end yaw
# c: curvature
def dubins_path_planning_from_origin(ex, ey, eyaw, c):
  dx = ex
  dy = ey
  D = math.sqrt(dx ** 2.0 + dy ** 2.0) #The straight line distance that the car must travel
  d = D * c # Distance/turning_radius
  theta = mod2pi(math.atan2(dy, dx)) # The yaw of the end position
  alpha = mod2pi(0.0 - theta) # The opposite of the yaw of the end poistion
  beta = mod2pi(eyaw - theta) # The difference between the desired ending yaw position and the result of going straight towards it

  #-------------------------------------------------------------------
  # YOUR CODE HERE
  #
  # Loop through all 6 of the planners, asking each one to compute a path
  #   Each planner will return t,p,q and mode
  #   t is the (signed) arc length of the first portion of the path, 
  #   p is (signed) arc length of the second portion, q is (signed) arc length of the third portion
  #   mode indicates which planner the path came from
  # Find the planner that returns the path with the smallest total arc length, (abs(t) + abs(p) + abs(q))
  # Set best_t,best_p,best_q, and best_mode to the t,p,q,and mode returned by the best planner
  #--------------------------------------------------------------------

  best_t, best_p, best_q, best_mode = None, None, None, None
  bcost = float('inf')
  planner_list = [LSL, RSR, LSR, RSL, RLR, LRL]

  for planner in planner_list:
    t, p, q, mode = planner(alpha, beta, d)
    if t == None or p == None or q == None:
      continue
    arc_distance = np.abs(t) + np.abs(p) + np.abs(q)
    if arc_distance < bcost:
      bcost = arc_distance
      best_t, best_p, best_q, best_mode = t, p, q, mode

  px, py, pyaw = generate_course([best_t, best_p, best_q], best_mode, c) # Turns arc lengths into points along path

  return px, py, pyaw, best_mode, bcost

def dubins_path_planning(s, e, c):
  """
  Dubins path plannner

  input:
      sx x position of start point [m]
      sy y position of start point [m]
      syaw yaw angle of start point [rad]
      ex x position of end point [m]
      ey y position of end point [m]
      eyaw yaw angle of end point [rad]
      c curvature [1/m]

  output:
      px
      py
      pyaw
      mode

  """
  
  sx, sy, syaw = s[0], s[1], s[2]
  ex, ey, eyaw = e[0], e[1], e[2]

  # Get path in frame of the source
  ex = ex - sx
  ey = ey - sy
  lex = math.cos(syaw) * ex + math.sin(syaw) * ey # Note that we are effectively rotating by -syaw
  ley = - math.sin(syaw) * ex + math.cos(syaw) * ey # Note that we are effectively rotating by -syaw
  leyaw = eyaw - syaw

  # Get the plan (w.r.t the source frame)
  lpx, lpy, lpyaw, mode, clen = dubins_path_planning_from_origin(lex, ley, leyaw, c)

  # Convert back to world coordinates
  px = [math.cos(-syaw) * x + math.sin(-syaw) * # Note that we are effectively rotating by syaw
        y + sx for x, y in zip(lpx, lpy)]
  py = [- math.sin(-syaw) * x + math.cos(-syaw) * # Note that we are effectively rotating by syaw
        y + sy for x, y in zip(lpx, lpy)]
  pyaw = [pi_2_pi(iyaw + syaw) for iyaw in lpyaw]

  # Filter out loops from path
  ppx, ppy, ppyaw, pclen = process_dubins(sx, sy, syaw, px, py, pyaw, clen)

  return ppx, ppy, ppyaw, pclen

def path_length(s, e, c):
  px, py, pyaw, cost = dubins_path_planning(s, e, c)
  return cost

def generate_course(length, mode, c):

  px = [0.0]
  py = [0.0]
  pyaw = [0.0]

  for m, l in zip(mode, length):
    pd = 0.0
    if m is "S":
      d = 0.3 * c
    else:  # turning couse
      d = math.radians(3.0)

    while pd < abs(l - d):
      #  print(pd, l)
      px.append(px[-1] + d / c * math.cos(pyaw[-1]))
      py.append(py[-1] + d / c * math.sin(pyaw[-1]))

      if m is "L":  # left turn
        pyaw.append(pyaw[-1] + d)
      elif m is "S":  # Straight
        pyaw.append(pyaw[-1])
      elif m is "R":  # right turn
        pyaw.append(pyaw[-1] - d)
      pd += d
    else:
      d = l - pd
      px.append(px[-1] + d / c * math.cos(pyaw[-1]))
      py.append(py[-1] + d / c * math.sin(pyaw[-1]))

      if m is "L":  # left turn
        pyaw.append(pyaw[-1] + d)
      elif m is "S":  # Straight
        pyaw.append(pyaw[-1])
      elif m is "R":  # right turn
        pyaw.append(pyaw[-1] - d)
      pd += d

  return px, py, pyaw

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
  """
  Plot arrow
  """

  if not isinstance(x, float):
    for (ix, iy, iyaw) in zip(x, y, yaw):
      plot_arrow(ix, iy, iyaw)
  else:
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
    plt.plot(x, y)

# Filter out loops from path
def process_dubins(startx, starty, enda, px, py, pa, cost):

  '''
  Naive processing of the output
  Ensuring there are no 2pi rotations due to numerical issues
  '''
  pcost = cost
  eps = 1e-6
  for i in range(1,len(px)-1):
    check1 = abs(px[i] - startx) < eps
    check2 = abs(py[i] - starty) < eps
    check3 = abs(pa[i] - enda) < eps
    if check1 and check2 and check3:
      return px[i:], py[i:], pa[i:], cost - math.radians(360.0)

  return px, py, pa, cost

def main():
  # Write TEST CODE HERE!

  #Random values for ex, ey, eyaw, c
  ex=1
  ey=1
  eyaw=np.pi/4
  c=np.pi/2

  lpx, lpy, lpyaw, mode, clen =dubins_path_planning_from_origin(ex, ey, eyaw, c)
  print('dubins result',lpx, lpy, lpyaw, mode, clen)

if __name__ == '__main__':
  main()
