	

import cv2
import requests

video_src = 'match5-c0.avi'
# video_src = 'video1.avi'
# video_src = 'parkinglot_1_480p.mp4'
# video_src = 'capturacars1.mp4'

# AVSS_PV_Medium_Divx.avi

# https://thingspeak.com/channels/278837
# https://data.sparkfun.com/input/E3DQO5Q9ryiraVNznRz3?private_key=d1kRPeRzYrTN5gjxbKxZ&cars=6.63&people=5.79
# sparck private key d1kRPeRzYrTN5gjxbKxZ
def puthMetric(cantidad):
    # url = "https://data.sparkfun.com/input/E3DQO5Q9ryiraVNznRz3?private_key=d1kRPeRzYrTN5gjxbKxZ&cars="+str(cantidad)+"&people=5.79"
    url = "https://api.thingspeak.com/update?api_key=HC6V2ZLVFEZP0R7X&field1="+str(cantidad)
    try:
        r = requests.get(url)
        print r
    except requests.ConnectionError:
        print("failed to connect")        
    


backsub = cv2.BackgroundSubtractorMOG()
capture = cv2.VideoCapture(video_src)
best_id=0
i = 0
ret = True
if capture:
  while ret:
    i = i+1
    
    ret, frame = capture.read()
    if ret:
        fgmask = backsub.apply(frame, None,0.01)
        contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        try: 
            hierarchy = hierarchy[0]
        except:
            print "error"
            hierarchy = []
        cantidad = 0
        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            if w > 20 and h > 20:
                cantidad = cantidad+1
                # figure out id
                # best_id+=1
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                if (i % 15 ==0):
                    print "cant {0}".format(cantidad)
                    puthMetric(cantidad)

                # cv2.putText(frame, str(best_id), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)
        # print(best_id)        
        cv2.imshow("Track", frame)
        cv2.imshow("background sub", fgmask)
    
    if cv2.waitKey(33) == 27:
        break
