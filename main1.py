import cv2

# Method 2
vidCap = cv2.VideoCapture('video2.mp4')

# initilize OpenCV - Background Subtractor for KNN and MOG2
BS_KNN = cv2.createBackgroundSubtractorKNN()
BS_MOG2 = cv2.createBackgroundSubtractorMOG2()

vehile = 0
validVehiles = []
while vidCap.isOpened():
    ret, frame = vidCap.read() # reads the next frame

    # extract the foreground mask
    fgMask = BS_MOG2.apply(frame)
    
    # draw the reference traffic lines
    cv2.line(frame, (350,400), (1500,400), (0,0,255), 2) # RED Line
    cv2.line(frame, (350,390), (1500,390), (0,255,0), 1) # GREEN Offset ABOVE
    cv2.line(frame, (350,410), (1500,410), (0,255,0), 1) # GREEN Offset BELOW
    
    # extract the contours
    conts, _ = cv2.findContours(fgMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in conts:
        x, y, w, h = cv2.boundingRect(c)
        
        # ignore the small contours in size
        visibleVehile = (w > 40) and (h > 40)
        if not visibleVehile:
            continue
        
        # remove the distraction on the road; consider only the objects on ROAD
        if x > 400 and x < 1300 and y > 200:
            # draw the bounding rectangle for all contours
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            xMid = int((x + (x+w))/2)
            yMid = int((y + (y+h))/2)
            cv2.circle(frame, (xMid,yMid),5,(0,0,255),5)

            # add all valid vehiles into List Array
            validVehiles.append((xMid,yMid))
#             cv2.waitKey(0) # debugging purpose

            for (vX, vY) in validVehiles:
                if vY > 690 and vY < 710: # adjust this for the frame jumping
                    vehile += 1
                    validVehiles.remove((vX,vY))
                    print("vehicle is detected : "+str(vehile))
                    
                    # debugging purpose
#                     cv2.putText(frame, 'Y : {}'.format(yMid), (x, y-20), cv2.Hershey Triplex, 2, (255,255,255), 2)

    # show the thresh and original video
    cv2.imshow('Foreground Mask', fgMask)
    cv2.putText(frame, 'Total Vehicles : {}'.format(vehile), (450, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255), 2)
    cv2.imshow('Original Video', frame)
    
    # wait for any key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('v'):
        break

# release video capture
cv2.destroyAllWindows()
vidCap.release()
