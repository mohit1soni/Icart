import serial
import keyboard
try:
    arduino=serial.Serial("COM8",timeout=1)
except:
    print("Please Check the port")

# raw_data=[]
count=1
wanted_data='i'
while True:
    raw_imu_data=[]
    raw_gps_data=[]
    if(count==1):
        wanted_data='g'
        arduino.write(wanted_data.encode())
    elif(count==0):
        wanted_data='i'
        arduino.write(wanted_data.encode())
    k=str(arduino.readline()).strip("b'").strip("\\r\\n")
    # print(k)
    if k == 'i':
        raw_imu_data.append(str(arduino.readline()).strip("'b").strip("\\r\\n").split(" "))
        print(raw_imu_data)
    if k == 'g':
        raw_gps_data.append(str(arduino.readline()).strip("'b").strip("\\r\\n").split(" "))
        print(raw_gps_data)
    count=int((count+1)%2)
    # print(k)
    if keyboard.is_pressed('a'):
        break