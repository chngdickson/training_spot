import sys
from controller import Supervisor, Robot
from controller import Motor,DistanceSensor, Gyro, Camera

#############  Initialization  ###############

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# Initialize robot
robot_name = "MY_ROBOT"
robot = supervisor.getFromDef(robot_name)
if robot is None:
    sys.stderr.write(f"No DEF {robot_name} node found in the current world file\n")
    sys.exit(1)

# Initialize robot sensors
motor_names = [
    "front left shoulder abduction motor",  "front left shoulder rotation motor",  "front left elbow motor",
    "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor",
    "rear left shoulder abduction motor",   "rear left shoulder rotation motor",   "rear left elbow motor",
    "rear right shoulder abduction motor",  "rear right shoulder rotation motor",  "rear right elbow motor"]

camera_names = [
    "left head camera", "right head camera", 
    "left flank camera","right flank camera", 
    "rear camera"]
gps_names = ["gps"]
gyro_names = ["gyro"]

def initialize_sensors(robot,device_names:list, enable_func_exist=True):
    sensors = []
    for i in range(len(device_names)):
        sensors.append(robot.getDevice(device_names[i]))
        if enable_func_exist:
            sensors[i].enable(timestep)
    return sensors

cameras = initialize_sensors(supervisor, camera_names)
gps = initialize_sensors(supervisor, gps_names)
gyros = initialize_sensors(supervisor, gyro_names)
#motors = initialize_sensors(supervisor, motor_names, False)

def initialize_motors_as_torques(motors):
    for motor in motors:
        #motor.setTorque(0)
        motor.enableTorqueFeedback(timestep)

#initialize_motors_as_torques(motors)
#################################################################
def get_pos(gps):
    # GPS gives us Position
    return gps[0].getValues()

def get_gyro(gyros):
    # Gyro provides us with velocities
    return gyros[0].getValues()
    
def get_camera(cameras, indices:list=None):
    rtn = []
    if indices is None:
        indices = [i for i in range(len(cameras))]
    for i in indices:
        camera = cameras[i]
        cameraData = camera.getImage()
        gray = Camera.imageGetGray(cameraData, camera.getWidth(),5,10)
        rtn.append(gray)
    return rtn
def get_motor_feedback(motors):
    feedback = []
    for motor in motors:
        feedback.append(motor.getTorqueFeedback())
    return feedback
#cameraData = cameras[0].getImage()
#gray = Camera.imageGetGray(cameraData, cameras[0].getWidth(), 5, 10)
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while supervisor.step(timestep) != -1:

    camera_imgs = get_camera(cameras,[2])
    pos = get_pos(gps)
    vel = get_gyro(gyros)
    #print(pos)
    #print(get_motor_feedback(motors))
    if pos[0]> 1 or pos[1] < -0.5:
        supervisor.simulationReset()
        robot.restartController()
        print("restarting_Simulation")
    pass

# Enter here exit cleanup code.
