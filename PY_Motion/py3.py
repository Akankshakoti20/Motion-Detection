import cv2, time, pandas
from datetime import datetime  
first_frame=None
status_list=[None,None] #before starting s1 is[0,0],[0,1][1,1]
times=[]
df=pandas.DataFrame(columns=["Start","End"])
video=cv2.VideoCapture(0)#This is a line where the web camera starts caputuring a video
while True:
    check,frame=video.read() #color frame gets stored here
    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #converting color
    gray=cv2.GaussianBlur(gray,(21,21),0) #caputuring the frame for the first time

    if first_frame is None:
        first_frame=gray #captured frame gats stored here
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame,27,255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=5)

    cnts,hierarchy=cv2.findContours(thresh_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<20000:
            continue
        status=1

        (x,y,w,h)=cv2.boundingRect(contour) #x,y,w,h stores coordinate values of the green rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        status_list.append(status) #[None,1]
        status_list=status_list[-2:] #negative indexing
         #start motion
        if status_list[-1]==1 and status_list[-2]==0:
            times.append(datetime.now())
        #end motion
        if status_list[-1]==0 and status_list[-2]==1:
            times.append(datetime.now())

        cv2.imshow("Gray Frame",gray)
        cv2.imshow("Delta",delta_frame)
        cv2.imshow("Threshold",thresh_frame)
        cv2.imshow("color",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
print(status_list)
print(times)

try:
    for i in range(0,len(times),2):
        df=df.append({"Start":times[i],"END":times[i+1]},ignore_index=True)
except:
    print("No time interval found")

df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows() 