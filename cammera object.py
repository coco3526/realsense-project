import pyrealsense2 as rs               #removed inputs
import numpy as np
from time import clock as timer
import cv2
import threading
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image
from playsound import playsound                                                           #for sound try see if can detect os for how sounds played by playsound or raspberry pi buzzer. same for tensorflow.



class realsenseBackbone():                                                                                                                          
    """
    Handels the backbone of the camera setting the configs and 
    """
    def __init__(self):
        #self.frames = rs.frames
        self.pipeline = rs.pipeline()
        self.config = self.setConfig()
        self.profile = self.pipeline.start(self.config)
        self.threedpoint = []
        
    def setConfig(self, bagfile = ""):
        #sets the config setting for the cammera
        #@param bagfile setsname of the file to read from send nothing to read form cammmera.
        config = rs.config()
        if len(bagfile) == 0:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            #config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
            #config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        else:
            config.enable_device_from_file(bagfile)    #can do ,false to not repeat
        return config

    def getpipeline(self):
        return self.pipeline

    def getConfig(self):
        return self.config
    
    def getFrames(self):
        #gets the fram from camera containg all the set stream
        frames = self.pipeline.wait_for_frames()
        return frames

    def getDepthFrame(self, frame):
        # input the frame thats collected
        #return depth frames
        depthFrame = frame.get_depth_frame()
        return depthFrame
    def getAlignedFrame(self, frame):
        """
        Align the depth frame to the color frame
        @return returns a aligned depth and color frame
        """
        #align_to is the stream type the frames are aligned with.
        align_to = rs.stream.color
        #rs.align is function used to align the frames.
        align = rs.align(align_to)
        aligned_frames = align.process(frames)
        return aligned_frames

    def getColorFrame(self, frame):
        # inputS the frame thats collected and returns color frame part.
         colorFrame = frame.get_color_frame()
         return colorFrame

    def colorImageCV2(self, frame):
         #puts the color image into a numpy array for cv2 display.
         colorFrame = self.getColorFrame(frame)
         colorImage = np.asanyarray(colorFrame.get_data())
         return colorImage

    def depthImageCV2(self, depthFrame):                                                                #think about usign opencv colorizer might make it more clear
        #inputs the depth frame
        #then apply color mapping to the image into a numpyarray to be displayed in cv2
        colorized = rs.colorizer(3) #can change vaule for color map
        colorized_depth = np.asanyarray(colorized.colorize(depthFrame).get_data())
        return colorized_depth


    """
    distance and 3d point realtive to the cammera functions
    """

    def distancePixel(self, depthFrame, x, y):
        """
        #takes in a depth frame form the camera and cordinates of the pixel 
        #from where the depth is wanted.
        @param depthFrame isa depth frame form the cammera without any filters applied.
        @param x is the x coridinate of the pixel
        @param y is the y cordinate of the pixel
        @param deproject retrun the depth in meters
        @retrun returns distance in meters
        """
        distance = depthFrame.get_distance(x, y)
        return distance

    def threePoint(self,depth_frame,x,y):
        """
        Determines the realtive 3D position of the cordinate from the pesepctive of the cammera.
        @param x coordinate of the pixel
        @param y coordinate of the pixel
        @param depth_frame retrived by the cammera to get the distance at the pixel must have no filters applied
        @retrun returns the relative postiton
        """
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_intrins = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        depth = self.distancePixel(depth_frame,x,y)
        deproject = rs.rs2_deproject_pixel_to_point( depth_intrins,[x, y], depth)
        return deproject

    def point3DContour(self,cnts,depth_frame):
        """
        apply threading to processs the lsit of points
        returns a list of points
        may need to do a lock
        send in a quue that can be addde on to or a list

        get rid of first for loop and append the the list sent in
        """
        threads = list() #create a lsit of threads
        self.threedpoint = [] #reset the list to be empty
        for i in cnts:
            thread = threading.Thread(target = self.process3DContour, args = (i, depth_frame))
            threads.append(thread)# add the thread to the list
            thread.start() #starst the tread.

        for j in threads:
            j.join() #join the thread causes a threa to wait till its finished
        return self.threedpoint

    def process3DContour(self, cnts, depth_frame):
        """
        Determines the 3d point of portions of th econtours no set point that is being loked at.
        @param cnts array of all of the positinos of the contours applied to the color image
        @param depth_frame must be a depth frame form cam without any filters applied
        @retrun returns a list of 3Dpoint in the contours
        """
        count = 0;
        for j in cnts:
            k = 0
            if (count%30 == 0):     #possibly bavk 10
                for k in j:
                    x = k[0]
                    y = k[1]
                    point = backbone.threePoint(depth_frame,x, y)
                    self.threedpoint.append(point)
            count = count + 1

    """
    Writeing to a file
    """
    def openfile(self, name):
        outf = open(name, "w")
        return outf

    def fileOutput(self,cnts, outf):
        for i in cnts:
            outf.write(str(i))
            outf.write("\n")
        return outf



    """
    Filters settings.
    must call depthimageCv2 to display any filters
    To get decimation to work must apply after hole filing not before.
    """
    def decimation(self, frame):
        #apply decimation filter to the frame that is sent in. Downscaling
        #must call depthImageCV2 to display

        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)
        decimated_depth = decimation.process(frame)
        return decimated_depth

    def spatial(self, frame):
        #apply spatial filtering to the frame sent in 
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, .25)
        spatial.set_option(rs.option.filter_smooth_delta, 50)
        spatial.set_option(rs.option.holes_fill, 1)
        filtered_depth = spatial.process(frame)
        return filtered_depth

    def hole(self, frame):
        hole_filling = rs.hole_filling_filter()
        holeFilling = hole_filling.process(depth_frame)
        return holeFilling

    def threshold(self, frame, minDistance, maxDistance):
        #apply threshold filter to a depth image from the min distance to max distance to be viewed
        threshold_filter = rs.threshold_filter(minDistance, maxDistance)
        return threshold_filter.process(frame)

    def disparity(self, frame):
        """
        converts the depth frame to disparity
        """
        disparity = rs.disparity_transform(True)
        depthFrame = disparity.process(frame)
        return depthFrame


    def allFilters(self, depthFrame, minDist, maxDist):                                                                                 #comment out filters not necessary 
        """
        Takes in a deoth fraam and applys all the filters to excludes decimation for perfromance but can be uncommented
        Depeneding on when you apply disparity it willl cause the threshold to not work apply after doing threshold
        @param minDist maxDist the distance the threshold filter is applied to.
        """
        #depthFrame = self.threshold(depthFrame, minDist, maxDist)
        depthFrame = self.disparity(depthFrame)
        depthFrame = self.spatial(depthFrame)
        #depthFrame = self.hole(depthFrame)
        #depthFrame = self.decimation(depthFrame)

        return depthFrame

    def contour(self, depthFrame, color_image):
        """
        Apply the contour to the collor image. by determining the threshold applied to the depth image and apply that outer edges to the color image as a contour
        @param depthFrame frame that has threshold filter applied
        @param color_image color image feed thats in a numpy array
        @return cnts retruns a the postion of the contours in an array of [[[]]]
        """

        low = np.array([110, 110, 110])
        high = np.array([150, 150, 150])
        """
        In range deterine if the inputed array is within the range of the other two array
        and outputs an array
        """
        mask = cv2.inRange(depthFrame, low, high)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(color_image, cnts,-1, [0,255,0], 2)
        return cnts


        

def loadLabels(labelfile):
    #load and return the label for the model
    labels = []
    with open(labelfile, "r", encoding = 'utf-8') as f:
        for line in f.readlines():
            labels.append(line)
    return labels


def object_distance(frame, min, max, realsensebackbone, default_distance = 0.5):
    #sends in the pixel coordinates of a object and the depth frame. Using the depth feed checks to see if a object
    #is to close to the cammera by traversing the point and retrieveing the disatnce form the cammera.
    xcoord = min[0]
    ycoord = min[1]
    xmax = max[0]
    ymax = max[1]

    xincrement = int((xmax - xcoord) / 5) #based on what you divided by will give you the number of points looked at.
    yincrement = int((ymax - ycoord) / 5)
    object_warn = False
    #traverse the x cordiantes stops if reaches end of object or object is detected.
    while (xcoord < xmax and object_warn != True):
        ycoord = min[1]
        #traverse the y coordinatess
        while (ycoord < ymax and object_warn != True):
            dist = realsensebackbone.distancePixel(frame, xcoord, ycoord)
            #if the object detectedd is to close warn the user.
            if (dist < default_distance and dist != 0):
                object_warn = True
            ycoord = ycoord + yincrement
        xcoord = xcoord + xincrement
    if (object_warn):
        playsound('Buzzer.mp3', False)



backbone = realsenseBackbone()
pipeline = backbone.getpipeline()
modelfile = "detect.tflite"
labelFile = "labelmap.txt"



if __name__ == "__main__":
    #load the model and allocate tensor
    interpreter  = Interpreter("detect.tflite")
    interpreter.allocate_tensors()
    #load labels of the model
    label = loadLabels(labelFile)
    #retrives the models input and output details
    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()

    #retrive the shape of image used to train the model
    height = inputDetails[0]['shape'][1]
    width = inputDetails[0]['shape'][2]

                                                               #may need to deal with if model is floating

    while (True):
        #playsound('Buzzer.mp3')
        #retrives the respective frames required and sends them where needed.
        start = timer()
        #retives the frames for the enabled streams from the camera and, aligns depth and color frame.
        frames = backbone.getFrames() #get the frame from the camera
        frames = backbone.getAlignedFrame(frames)

        timeStamp = frames.get_timestamp() / 1000
        #retrives the depth image from camera
        depth_frame = backbone.getDepthFrame(frames)

        # retrives color image as a np array
        color_image = backbone.colorImageCV2(frames)
        origH = color_image.shape[0]
        origW = color_image.shape[1]

        #reize color image to be same as model 
        #make the image shape be [1 x height x width x 3] 
        #3 correlates to rgb format. 
        color_RGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) #convert the color to rgb
        color_resize =cv2.resize(color_RGB, (width, height))
        image_data = np.expand_dims(color_resize, axis=0) #expand the diminstion to 4 making first one 1 item
        
        #perform detection on the resized image
        interpreter.set_tensor(inputDetails[0]['index'], image_data)
        interpreter.invoke()

        #threshold correlating to the probability the calss was detected
        threshold = .40

        #retrieve the model output first index correlates to output array
        boxes = interpreter.get_tensor(outputDetails[0]['index'])[0] # Bounding box cordinates                                                                #understand get tensor
        classes = interpreter.get_tensor(outputDetails[1]["index"])[0] # Class index of object/label
        score = interpreter.get_tensor(outputDetails[2]["index"])[0] # Score/ prediction a class was detected or confidence
        count = int(interpreter.get_tensor(outputDetails[3]["index"])[0]) # Number of objects detected 

        """Retrieve the bounding box and scale it to the original image
        draw the boundiung box onto the image and apply the label above the object
        """
        for i in range(count):                                                                                                                      #check if count is correct.
            if(score[i] >= threshold):
                #retive the bounding box coordinates. Must make sure that the coordinates are not ouside the image 
                ymin = int(max(1,(boxes[i][0] * origH)))
                xmin = int(max(1, (boxes[i][1] * origW)))
                ymax = int(min(origH, (boxes[i][2] * origH)))
                xmax = int(min(origW, (boxes[i][3] * origW)))
                object_distance(depth_frame, (xmin,ymin), (xmax,ymax), backbone)
                #print(ymin, xmin, ymax, xmax)
                #draws the boundiung box
                cv2.rectangle(color_image, (xmin,ymin), (xmax,ymax), (10,255,0), 4, cv2.LINE_4)

                #apply the label to each object
                #print(classes[i])
                if(int(classes[i]) < len(label)):
                    object_name = label[int(classes[i])] # Look up the object in the label array from the class index
                    label_name =  '%s: %d%%' % (object_name, int(score[i]*100))
                    #labelSize, baseLine = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    #label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window

                    #cv2.rectangle(color_image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in

                    #cv2.putText(color_image, label_name, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    cv2.putText(color_image, label_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2) 

            


        #apply all the filters to the depth and converts it to np.array
        depthFrame = backbone.allFilters(depth_frame, 2, 4)
        depthFrame = backbone.depthImageCV2(depthFrame)


        #frame display
        FPS = "{:.2f}".format(1 / (timer() - start))
        cv2.putText(color_image, FPS, (10,15), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,0,0), 1, cv2.LINE_AA)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        res = np.hstack((depthFrame, color_image))
        cv2.imshow('RealSense', res)

        #end the program when the windows is closed
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
        

pipeline.stop()

