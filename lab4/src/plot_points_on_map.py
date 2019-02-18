#!/usr/bin/env python

import numpy as np
import sys
import cv2

CIRCLE_RADIUS = 20

def main():
  argv = sys.argv
  if len(argv) != 5:
    print 'Usage: python plot_points_on_map.py <map_image.pgm> <start.csv> <good.csv> <bad.csv>'
    exit()

  _, map_file, start_file, good, bad = sys.argv

  map_image = cv2.imread(map_file, cv2.IMREAD_COLOR)
  start_points = np.loadtxt(start_file, delimiter=',', skiprows=1, dtype=np.int, ndmin=2)
  good_points = np.loadtxt(good, delimiter=',', skiprows=1, dtype=np.int, ndmin=2)
  bad_points  = np.loadtxt(bad, delimiter=',', skiprows=1, dtype=np.int, ndmin=2)

  for x, y in bad_points:
    cv2.circle(map_image, (x, y), CIRCLE_RADIUS, color=(0,0,255), thickness=-1)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(map_image,'({},{})'.format(x,y),(x, y), font, fontScale=2, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)

  for x,y in start_points:
    cv2.circle(map_image, (x, y), CIRCLE_RADIUS, color=(0,255,0), thickness=-1)

  for i, (x, y) in enumerate(good_points):
    cv2.circle(map_image, (x, y), CIRCLE_RADIUS, color=(255,0,0), thickness=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(map_image,'{}({},{})'.format(i,x,y),(x, y), font, fontScale=2, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)

  def echo(event,x,y,flags,param):
      global mouseX,mouseY
      if event == cv2.EVENT_LBUTTONDBLCLK:
          print '{},{}'.format(x, y)

  cv2.namedWindow('map', cv2.WINDOW_NORMAL)
  cv2.setMouseCallback('map',echo)
# 
  # while True:
  cv2.imshow('map',map_image)
  k = cv2.waitKey(0) & 0xFF
  # if k == 27:
          # break

  # cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()