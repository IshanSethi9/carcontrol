#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries
import cv2
import numpy as np
import pyautogui
import keyboard


# In[2]:


#Color to detect BGR
l = [17, 15, 100] #lower
u = [80, 76, 220] #upper


# In[3]:


#region coordinates 
k_left, k_top, k_right, k_bottom = 640, 30, 440, 130
h_left, h_top, h_right, h_bottom = 440, 130, 240, 330
s_left, s_top, s_right, s_bottom = 840, 130, 640, 330
f_left, f_top, f_right, f_bottom = 640, 330, 440, 430


# In[4]:


#Key Pressed
current_key_pressed = set()


# In[5]:


#Accelerate
def up():
    #print("W")
    pyautogui.keyDown('up')
    current_key_pressed.add('w')


# In[6]:


#Steering Right
def right():
    #print("D")
    pyautogui.keyDown('right')
    current_key_pressed.add('d')


# In[7]:


#Steering Left
def left():
    #print("A")
    pyautogui.keyDown('left')
    current_key_pressed.add('a')


# In[8]:


#Brakes
def down():
    #print("S")
    pyautogui.keyDown('down')
    current_key_pressed.add('s')


# In[9]:


#Find contours
def findContours(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)[1]
    (_, cnts, _) = cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return len(cnts)


# In[10]:


#Main function
if __name__=='__main__':
    aWeight=0.5

    cam=cv2.VideoCapture(0)

    cam.set(3,1280)
    cam.set(4,720)
    cam.set(cv2.CAP_PROP_FPS,60)

    while True:
        buttonPressed = False
        buttonPressed_leftright = False
        
        status, frame = cam.read()

        clone = frame.copy()
        clone = cv2.flip(clone,1)
        clone = cv2.resize(clone,(1280,720))

        reg_up = clone[k_top:k_bottom, k_right:k_left]
        reg_left = clone[h_top:h_bottom, h_right:h_left]
        reg_right = clone[s_top:s_bottom, s_right:s_left]
        reg_down = clone[f_top:f_bottom, f_right:f_left]

        reg_up = cv2.GaussianBlur(reg_up, (7,7), 0)
        reg_right = cv2.GaussianBlur(reg_right, (7,7), 0)
        reg_left = cv2.GaussianBlur(reg_left, (7,7), 0)
        reg_down = cv2.GaussianBlur(reg_down, (7,7), 0)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, l, u)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        l = np.array(lower, dtype="uint8")
        u = np.array(upper, dtype="uint8")

        mask_up = cv2.inRange(reg_up, l, u)
        mask_right = cv2.inRange(reg_right, l, u)
        mask_left = cv2.inRange(reg_left, l, u)
        mask_down = cv2.inRange(reg_down, l, u)

        out_up = cv2.bitwise_and(reg_up, reg_up, mask=mask_up)
        out_right = cv2.bitwise_and(reg_right, reg_right, mask=mask_right)
        out_left = cv2.bitwise_and(reg_left, reg_left, mask=mask_left)
        out_down = cv2.bitwise_and(reg_down, reg_down, mask=mask_down)

        cnts_up = findContours(out_up)
        cnts_right = findContours(out_right)
        cnts_left = findContours(out_left)
        cnts_down = findContours(out_down)

        if (cnts_up > 0):
            up()
            buttonPressed = True
      
        elif (cnts_right > 0):
            right()
            buttonPressed = True
            buttonPressed_leftright = True

        elif (cnts_left > 0):
            left()
            buttonPressed = True
            buttonPressed_leftright = True
        
        elif (cnts_down > 0):
            down()
            buttonPressed = True

        image_up = cv2.rectangle(clone, (k_left, k_top), (k_right, k_bottom), (255,0,255,0.5), 2)
        image_left = cv2.rectangle(clone, (h_left, h_top), (h_right, h_bottom), (255,0,0,0.5), 2)
        image_right = cv2.rectangle(clone, (s_left, s_top), (s_right, s_bottom), (0,0,255,0.5), 2)
        image_down = cv2.rectangle(clone, (f_left, f_top), (f_right, f_bottom), (0,255,255,0.5), 2)
      
        cv2.putText(image_up, "W", (k_left-170,k_top+110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(image_left, "A", (h_left-170,h_top+200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(image_right, "D", (s_left-170,s_top+200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(image_down, "S", (f_left-170,f_top+110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.namedWindow("video",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("video", clone)

        if not buttonPressed and len(current_key_pressed) != 0:
            for key in current_key_pressed:
                pyautogui.keyUp(key)
            current_key_pressed = set()
            
        if not buttonPressed_leftright and (('a' in current_key_pressed) or ('d' in current_key_pressed)): 
            if 'a' in current_key_pressed:
                pyautogui.keyUp('left')
                current_key_pressed.remove('a')  
            elif 'd' in current_key_pressed:
                pyautogui.keyUp('right')
                current_key_pressed.remove('d')

        if cv2.waitKey(1) & 0Xff == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

