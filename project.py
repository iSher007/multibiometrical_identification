from time import sleep
import time
import RPi.GPIO as GPIO
import smbus2 as smbus
from picamera import PiCamera
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import serial
import adafruit_fingerprint
from mfrc522 import SimpleMFRC522

data_path = '/home/pi/Desktop/Dissertation_Project/faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

face_classifier = cv2.CascadeClassifier('/home/pi/Desktop/Dissertation_Project/data/haarcascade_frontalface_default.xml')

# uart = serial.Serial("/dev/ttyUSB0", baudrate=57600, timeout=1)
uart = serial.Serial("/dev/ttyS0", baudrate=57600, timeout=1)

finger = adafruit_fingerprint.Adafruit_Fingerprint(uart)

GPIO.setmode(GPIO.BOARD)

redled = 40
greenled = 38

btn1 = 36
btn2 = 33
btn3 = 35
btn4 = 37

buzzer = 32

GPIO.setup(btn1, GPIO.IN)
GPIO.setup(btn2, GPIO.IN)
GPIO.setup(btn3, GPIO.IN)
GPIO.setup(btn4, GPIO.IN)

GPIO.setup(greenled, GPIO.OUT)
GPIO.setup(redled, GPIO.OUT)

GPIO.setup(buzzer, GPIO.OUT)

GPIO.output(greenled, GPIO.LOW)
GPIO.output(redled, GPIO.LOW)

GPIO.output(buzzer, GPIO.LOW)

I2C_ADDR = 0x3f
LCD_WIDTH = 16

LCD_CHR = 1
LCD_CMD = 0

LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0

LCD_BACKLIGHT = 0x08

ENABLE = 0b00000100

E_PULSE = 0.0005
E_DELAY = 0.0005

bus = smbus.SMBus(1)

#-------------Beginning of FACE RECOGNITION-------
def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

def face_detect_save(count):
    cap = cv2.VideoCapture(0)

    count2 = count + 3
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            if(count==count2):
                break
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = '/home/pi/Desktop/Dissertation_Project/faces/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            lcd_string("Face Found!", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
            pass

        if cv2.waitKey(1)==13:
            break

    cap.release()
    cv2.destroyAllWindows()
    lcd_string("Colleting Samples", LCD_LINE_1)
    lcd_string("Complete!!!", LCD_LINE_2)

def face_recognise():
    cv2.destroyAllWindows()
    sleep(1)
    cap = cv2.VideoCapture(0)
    confidence = 0
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'% Confidence it is user'
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


            if confidence > 75:
                cv2.putText(image, "Hi, User!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
                break

            else:
                cv2.putText(image, "Unknown Person", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1)==13:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    return confidence

#-------------END of FACE RECOGNITION-------------

#-------------Beginning of Fingerprint------------
def fingerprint_check():
    
    while True:
        
        a = "0 0"
        if finger.read_templates() != adafruit_fingerprint.OK:
            raise RuntimeError("Failed to read templates")
        aaa = get_fingerprint()
        if (aaa==True):
            a = str(finger.finger_id)+" "+str(int(finger.confidence*(100/255)))
            break
        elif(int(aaa)==15):
            lcd_string("Time is up!", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
            sleep(0.2)
            a = "time"
            break
        else:
            lcd_string("Finger Not Found!", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
            sleep(0.2)
            a = "0 0"
    return a

def fingerprint_enroll(count):
    count = count/3 + 1
    if finger.read_templates() != adafruit_fingerprint.OK:
        raise RuntimeError("Failed to read templates")
    a = 1
    enroll_finger(int(count))
    return a

def fingerprint_delete():
    while True:
        a = 0
        if finger.read_templates() != adafruit_fingerprint.OK:
            raise RuntimeError("Failed to read templates")
        if finger.delete_model(get_num()) == adafruit_fingerprint.OK:
            a = 1
            lcd_string("Deleted", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
            break
        else:
            a = 0
            lcd_string("Failed To Delete", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
    return a

def fingerprint_empty():
    
    if finger.read_templates() != adafruit_fingerprint.OK:
        raise RuntimeError("Failed to read templates")
    lcd_string("Deleting!", LCD_LINE_1)
    lcd_string("", LCD_LINE_2)
    for i in range(128):
        if finger.delete_model(i) == adafruit_fingerprint.OK:
            a = 1
        else:
            a = 0
            lcd_string("Failed To Delete", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)

def get_fingerprint():
    timeout = time.time() + 10
    """Get a finger print image, template it, and see if it matches!"""
    lcd_string("Waiting for", LCD_LINE_1)
    lcd_string("finger image...", LCD_LINE_2)
    while finger.get_image() != adafruit_fingerprint.OK:
        end_time = time.time()
        if((end_time > timeout)):
            return 15
        pass
    lcd_string("Templating...", LCD_LINE_1)
    lcd_string("", LCD_LINE_2)
    sleep(0.2)
    if finger.image_2_tz(1) != adafruit_fingerprint.OK:
        return False
    lcd_string("Searching...", LCD_LINE_1)
    lcd_string("", LCD_LINE_2)
    if finger.finger_fast_search() != adafruit_fingerprint.OK:
        return False
    return True
 
 
# pylint: disable=too-many-branches
def get_fingerprint_detail():
    """Get a finger print image, template it, and see if it matches!
    This time, print out each error instead of just returning on failure"""
    lcd_string("Getting Image...", LCD_LINE_1)
    lcd_string("", LCD_LINE_2)
    i = finger.get_image()
    if i == adafruit_fingerprint.OK:
        lcd_string("Image Taken", LCD_LINE_1)
        lcd_string("", LCD_LINE_2)
    else:
        if i == adafruit_fingerprint.NOFINGER:
            lcd_string("No finger", LCD_LINE_1)
            lcd_string("detected", LCD_LINE_2)
        elif i == adafruit_fingerprint.IMAGEFAIL:
            lcd_string("Image Error", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        else:
            lcd_string("Other Error", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        return False
 
    lcd_string("Templating...", LCD_LINE_1)
    lcd_string("", LCD_LINE_2)
    i = finger.image_2_tz(1)
    if i == adafruit_fingerprint.OK:
        lcd_string("Templated!", LCD_LINE_1)
        lcd_string("", LCD_LINE_2)
    else:
        if i == adafruit_fingerprint.IMAGEMESS:
            lcd_string("Image too", LCD_LINE_1)
            lcd_string("messy", LCD_LINE_2)
        elif i == adafruit_fingerprint.FEATUREFAIL:
            lcd_string("Couldn't identif", LCD_LINE_1)
            lcd_string("y features", LCD_LINE_2)
        elif i == adafruit_fingerprint.INVALIDIMAGE:
            lcd_string("Image invalid", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        else:
            lcd_string("Other Error", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        return False
 
    lcd_string("Searching", LCD_LINE_1)
    lcd_string("", LCD_LINE_2)
    i = finger.finger_fast_search()
    # pylint: disable=no-else-return
    # This block needs to be refactored when it can be tested.
    if i == adafruit_fingerprint.OK:
        lcd_string("Found Fingerprint", LCD_LINE_1)
        lcd_string("", LCD_LINE_2)
        return True
    else:
        if i == adafruit_fingerprint.NOTFOUND:
            lcd_string("No match found", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        else:
            lcd_string("Other Error", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        return False
 
 
# pylint: disable=too-many-statements
def enroll_finger(location):
    """Take a 2 finger images and template it, then store in 'location'"""
    for fingerimg in range(1, 3):
        if fingerimg == 1:
            lcd_string("Place Finger", LCD_LINE_1)
            lcd_string("on sensor...", LCD_LINE_2)
        else:
            lcd_string("Place same Finger", LCD_LINE_1)
            lcd_string("r again...", LCD_LINE_2)
 
        while True:
            i = finger.get_image()
            if i == adafruit_fingerprint.OK:
                lcd_string("Image Taken", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
                break
            if i == adafruit_fingerprint.NOFINGER:
                lcd_string("...", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
            elif i == adafruit_fingerprint.IMAGEFAIL:
                lcd_string("Image Error", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
                return False
            else:
                lcd_string("Other Error", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
                return False
 
        lcd_string("Templating...", LCD_LINE_1)
        lcd_string("", LCD_LINE_2)
        i = finger.image_2_tz(fingerimg)
        if i == adafruit_fingerprint.OK:
            lcd_string("Templated", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        else:
            if i == adafruit_fingerprint.IMAGEMESS:
                lcd_string("Image too Messy", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
            elif i == adafruit_fingerprint.FEATUREFAIL:
                lcd_string("Couldn't identif", LCD_LINE_1)
                lcd_string("y features", LCD_LINE_2)
            elif i == adafruit_fingerprint.INVALIDIMAGE:
                lcd_string("Image invalid", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
            else:
                lcd_string("Other Error", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
            return False
 
        if fingerimg == 1:
            lcd_string("Remove Finger", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
            sleep(1)
            while i != adafruit_fingerprint.NOFINGER:
                i = finger.get_image()
 
    lcd_string("Creating model...", LCD_LINE_1)
    lcd_string("", LCD_LINE_2)
    i = finger.create_model()
    if i == adafruit_fingerprint.OK:
        lcd_string("Created", LCD_LINE_1)
        lcd_string("", LCD_LINE_2)
    else:
        if i == adafruit_fingerprint.ENROLLMISMATCH:
            lcd_string("Prints did not ", LCD_LINE_1)
            lcd_string("match", LCD_LINE_2)
        else:
            lcd_string("Other Error", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        return False
 
    lcd_string("Storing model to", LCD_LINE_1)
    lcd_string(str(location)+"location", LCD_LINE_2)
    i = finger.store_model(location)
    if i == adafruit_fingerprint.OK:
        lcd_string("Stored", LCD_LINE_1)
        lcd_string("", LCD_LINE_2)
        return True
    else:
        if i == adafruit_fingerprint.BADLOCATION:
            lcd_string("Bad storage", LCD_LINE_1)
            lcd_string("location", LCD_LINE_2)
        elif i == adafruit_fingerprint.FLASHERR:
            lcd_string("Flash storage", LCD_LINE_1)
            lcd_string("error", LCD_LINE_2)
        else:
            lcd_string("Other Error", LCD_LINE_1)
            lcd_string("", LCD_LINE_2)
        return False
 

##################################################

def get_num():
    """Use input() to get a valid number from 1 to 127. Retry till success!"""
    i = 0
    while (i > 127) or (i < 1):
        try:
            i = int(input("Enter ID # from 1-127: "))
        except ValueError:
            pass
    return i
#-------------END of Fingerprint------------------

#-------------Beginning of RFID-------------------
#reading data from rfid tag
def readCard():
    c = ""
    cardReader = SimpleMFRC522()
    lcd_string("Scanning for", LCD_LINE_1)
    lcd_string("card...", LCD_LINE_2)
    #print ("to cancel press ctrl+c")
    id1, text = cardReader.read()
    if(id1 == 787545154257):
        c = "admin"
    else:
        c = "user"
    return c

#writing data to the rfid tag

def writeCard():
    d = 1
    cardWrite = SimpleMFRC522()
    text = "MultipleBiometricSystem"
    lcd_string("place your tag", LCD_LINE_1)
    lcd_string("to write: ", LCD_LINE_2)
    cardWrite.write(text)
    lcd_string("Write Successful", LCD_LINE_1)
    lcd_string("ly", LCD_LINE_2)
    return d

#-------------END of RFID-------------------------

#-------------Beginning of LCD--------------------
def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    sleep(E_DELAY)


def lcd_byte(bits, mode):
    bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
    bits_low = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
    
    bus.write_byte(I2C_ADDR, bits_high)
    lcd_toggle_enable(bits_high)

    bus.write_byte(I2C_ADDR, bits_low)
    lcd_toggle_enable(bits_low)


def lcd_toggle_enable(bits):
    sleep(E_DELAY)
    bus.write_byte(I2C_ADDR, (bits | ENABLE))
    sleep(E_PULSE)
    bus.write_byte(I2C_ADDR, (bits & ~ENABLE))
    sleep(E_DELAY)


def lcd_string(message, line):
    message = message.ljust(LCD_WIDTH, " ")

    lcd_byte(line, LCD_CMD)

    for i in range(LCD_WIDTH):
        lcd_byte(ord(message[i]), LCD_CHR)
        
#-------------End of LCD--------------------------

#-------------Beginning of Buzzer-----------------
def beep():
    GPIO.output(buzzer,GPIO.HIGH)
    sleep(0.05)
    GPIO.output(buzzer,GPIO.LOW)
#-------------End of Buzzer-----------------------

#-------------On Green LED------------------------
def green_on():
    GPIO.output(greenled,GPIO.HIGH)
#-------------------------------------------------

#-------------Off Green LED-----------------------
def green_off():
    GPIO.output(greenled,GPIO.LOW)
#-------------------------------------------------

#-------------On Red LED--------------------------
def red_on():
    GPIO.output(redled,GPIO.HIGH)
#-------------------------------------------------

#-------------Off Red LED-------------------------
def red_off():
    GPIO.output(redled,GPIO.LOW)
#-------------------------------------------------

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

lcd_string("Model Training", LCD_LINE_1)
lcd_string("Complete!!!!!", LCD_LINE_2)

count = len([item for item in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, item))])

if __name__ == '__main__':
    #fingerprint_empty()
    #fingerprint_enroll(0)
    lcd_init()
    lcd_string("Multi-Biometric", LCD_LINE_1)
    lcd_string("Identification!", LCD_LINE_2)
    permission_admin = False
    try:
        while True:
            go_away = False
            dtct = face_recognise()
            while(int(dtct)>75):
                timeout = time.time()+60
                while(permission_admin):
                    end_time = time.time()
                    if ( GPIO.input(btn1) == False ):
                        beep()
                        lcd_string("Adding New user!", LCD_LINE_1)
                        lcd_string("Face Recognising", LCD_LINE_2)
                        face_detect_save(count)

                    if ( GPIO.input(btn2) == False ):
                        beep()
                        lcd_string("New Fingerprint!", LCD_LINE_1)
                        lcd_string("*****Adding!*****", LCD_LINE_2)
                        fingerprint_enroll(count)

                    if ( GPIO.input(btn3) == False ):
                        beep()
                        lcd_string("New RFID!", LCD_LINE_1)
                        lcd_string("****Writing!****", LCD_LINE_2)
                        writeCard()
                        
                    if ( end_time>timeout or GPIO.input(btn4) == False ):
                        beep()
                        green_off()
                        permission_admin = False
                        sleep(1)
                        lcd_string("Mode Admin:Off!", LCD_LINE_1)
                        lcd_string("****************", LCD_LINE_2)
                        sleep(2)
                        red_off()
                        green_off()
                        lcd_string("Face Recognising!", LCD_LINE_1)
                        lcd_string("****************", LCD_LINE_2)
                        go_away = True
                if(go_away):
                    break
                
                lcd_string("Welcome!", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
                sleep(0.2)
                a = dtct
                lcd_string(str(a)+"% USER", LCD_LINE_1)
                lcd_string("Fingerprint Check!", LCD_LINE_2)
                sleep(0.2)
                b = str(fingerprint_check())
                if(b=="time"):
                    break
                else:
                    bb = b.split(" ")
                    lcd_string("Id "+str(bb[0])+" "+str(bb[1])+"% USER", LCD_LINE_1)
                    lcd_string("Tag RFID!", LCD_LINE_2)
                sleep(0.5)
                c = readCard()
                lcd_string("Read Successfully", LCD_LINE_1)
                lcd_string("", LCD_LINE_2)
                sleep(0.5)
                if(int(a)>75 and int(b[0])==1 and c=="admin"):
                    green_on()
                    red_off()
                    permission_admin = True
                    lcd_string("", LCD_LINE_1)
                    lcd_string("Welcome, Admin!", LCD_LINE_2)
                    sleep(2)
                elif(int(a)>75 and int(b[0])!=1 and c=="user"):
                    green_on()
                    red_off()
                    permission_admin = False
                    lcd_string("", LCD_LINE_1)
                    lcd_string("User found", LCD_LINE_2)
                    sleep(2)
                    red_off()
                    green_off()
                    lcd_string("Face Recognising!", LCD_LINE_1)
                    lcd_string("****************", LCD_LINE_2)
                    break
                else:
                    permission_admin = False
                    green_off()
                    red_on()
                    lcd_string("", LCD_LINE_1)
                    lcd_string("No Permission", LCD_LINE_2)
                    sleep(2)
                    red_off()
                    green_off()
                    lcd_string("Face Recognising!", LCD_LINE_1)
                    lcd_string("****************", LCD_LINE_2)
                    break
                sleep(0.1)
            sleep(0.1)
    except:
        GPIO.cleanup()
        lcd_init()

#timeout = time.time()+10
#while(True):
#  end_time = time.time()
#  if(end_time>timeout):
#    return 10, "time"
  
