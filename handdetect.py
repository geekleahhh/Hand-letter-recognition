#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.com/research/track/camshift.html

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# local module
import video
from video import presets


from video import create_capture
from common import clock, draw_str
from test import LetterStatModel
from test import MLP
from test import Boost
import sys, getopt


model_choose='boost'
model_name=model_choose

#detect face using adaboost
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                    flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0: 
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#draw rectangle in an image
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    samples, responses = a[:,1:], a[:,0]
    return samples, responses
       
class App(object):
    def __init__(self, args, video_src):
        self.cam = video.create_capture(video_src, presets['cube']) 
        cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
        cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
        while(True):  #capture the face until face is detected
            determine=self.captureFace(cascade) #capture face since and store it into determine
            if(determine!=[]): #Once capture the face, stop the forloop
                break
            if cv.waitKey(5) == 27:
                break
        cv.destroyAllWindows()
        cv.namedWindow('camshift')
        self.show_backproj = False #whether to backproj

    #Using adaboost to firstly capture the face
    def captureFace(self,cascade):
        _ret, img = self.cam.read()
        vis = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)
        if(rects!=[]):
            for xmin, ymin, xmax, ymax in rects:
                self.selection=(xmin,ymin,xmax,ymax)
                self.track_window = (xmin,ymin,xmax-xmin, ymax-ymin)
                self.handwindow = self.track_window    
        draw_rects(vis, rects, (0, 255, 0))
        cv.imshow('facedetect', vis)
        return rects

    #show the histogram of given area 
    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    #after capturing the face using adaboost, try to use camshift to follow the similar area based on color
    def run(self):
        loop=120
        while True:
            _ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV) # convert image from BGR to HSV
            mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.))) # find field in color np.array((0., 60., 32.)), np.array((180., 255., 255.)
            if self.selection:
                x0, y0, x1, y1 = self.selection
                #from the detect function
                hsv_roi = hsv[y0:y1, x0:x1] #change to hsv
                mask_roi = mask[y0:y1, x0:x1] #change to specific range color
                hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 380] ) #calculate the histogram
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX) #normalize the histogram
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv.calcBackProject([hsv], [0], self.hist, [0, 180], 1) #calculate the backProject probability image based on the histogram
                prob &= mask
                

                term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 ) 
                track_box, self.track_window = cv.CamShift(prob, self.track_window, term_crit) #calculate the most similar area using camshift and stored in trackbox and self.trackwindow
                xmin,ymin,w1,w2=self.track_window
                prob_black=prob
                #prob_black[ymin-w2/5:(ymin+w2*6/5),xmin-w1/5:(xmin+w1*6/5)]=0 #make the probability of face equal to 0
                prob_black[ymin:(ymin+w2),xmin:(xmin+w1)]=0
                track_box2, self.handwindow = cv.CamShift(prob_black, self.handwindow, term_crit) #find the most similar area in the image after removing the face, generally it will be your hand
                (x,y,w1,w2)=self.handwindow #store the hand's position
                x1=int(x)
                x2=int(x+w1*2)
                y1=int(y)
                y2=int(y+w2*6/5)
                try:
                    cv.ellipse(vis, track_box2, (0, 0, 255), 2) #draw the ellipse(of hand)
                    cv.rectangle(vis, (x1, y1), (x2, y2),(0,255,2), 2) #draw the rectangle(of hand)
                except:
                    print(rects_hand)
                rects_hand=[x,y,x+w1,y+w2] #rects_hand is used to draw the hand's triangle in the image
                #if want to see the backproject probability image
                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]   #vis store the probability image                 
                    jpg=vis[y1:y2,x1:x2]  #jpg store the hand part in the probability image
                    try:
                        img_dst=cv.resize(jpg,(16,16)) #resize the hand image
                        img_dst=np.reshape(np.array(img_dst),[-1,256])
                        print('succesful resize')
                        models = [MLP,Boost] # the models can be used to train
                        models = dict( [(cls.__name__.lower(), cls) for cls in models] )
                        args, dummy = getopt.getopt(sys.argv[1:], '', ['model=', 'data=', 'load=', 'save='])
                        args = dict(args)
                        args.setdefault('--model', model_choose)
                        Model = models[args['--model']]
                        sample=np.array(img_dst)  #transfer list to array
                        sample=np.float32(sample) 
                        #print(sample.shape)
                        #sample=np.reshape(sample,[-1,256])
                        #print(sample)
                        model = Model()
                        model.load(model_name+'.txt') #load the model we trained and saved in boost.txt and mlp.txt
                        print('testing...')
                        test_result  = model.predict(sample) #predict the letter of your hand, c or l
                        test_result = chr(test_result+97)
                        print('The word is : %s ' % test_result)
                        ch = cv.waitKey(5)
                        #if ch == ord('g'): #if you click'g', means you want to store your hand picture
                        loop=loop+1
                        cv.imwrite('/Users/yaoweili/Desktop/delete/letter/o'+str(loop)+'.jpg',img_dst)
                            #cv.imwrite('/Users/yaoweili/Desktop/delete/letter/c'+'bigger'+str(loop)+'.jpg',img_dst2)
                    except:
                        print('cannot resize')


            cv.imshow('camshift', vis)
            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj #if you click b, show the backprojection
        cv.destroyAllWindows()


if __name__ == '__main__':
    import sys, getopt
    print(__doc__)
    args, tt = getopt.getopt(sys.argv[1:], '', ['cascade='])
    args = dict(args)
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    print(__doc__)
    App(args,video_src).run()
