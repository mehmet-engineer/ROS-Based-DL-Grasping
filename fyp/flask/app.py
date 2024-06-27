#/usr/bin/env python3
from flask import Flask, render_template, Response, request, url_for

import rospy
from communication import ROSBackend 
import threading
import time
import sys
import cv2
import numpy as np
import base64
import datetime

app = Flask(__name__)
# run_with_ngrok(app)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
log = ""

# Ignore this part, this for clear the cache
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route('/', methods=["GET", "POST"])
def index():
    """Video streaming home page."""
    
    if request.method == "POST":
        if request.form.get('START') == "start":
            update_status("Connecting to robot")
            task_status = ros_server.send_start_command()
            update_status(f"{task_status}")
        
        if request.form.get('PREDICT') == "predict":
            update_status(space=True)
            update_status("Detecting grasps..")
            ros_server.predict_grasps()
            while ros_server.results is None: continue
            grasps_count = len(ros_server.grasps)
            update_status(f"[{grasps_count}] grasps predicted")
            
        if request.form.get('SELECT') == "select":
            selected_grasp = request.form.get('grasp_select')
            selected_grasp = int(selected_grasp)
            ros_server.curr_n_grasps.remove(selected_grasp)
            update_status(f"Grasp [{selected_grasp}] selected")
            task_status = ros_server.send_grasp_command(selected_grasp)
            update_status(f"{task_status}")
        
        if request.form.get('RESET_GRASP') == "reset_grasp":
            ros_server.reset_grasp()
            update_status("Grasps reset")
        
        if request.form.get('SEND') == "send_joint":
            joint_target = [ int(request.form.get(f"Joint_{i+1}")) for i in range(len(ros_server.current_joint))]
            update_status(space=True)
            update_status("Send joint command")
            task_status = ros_server.send_joint_command(joint_target)
            update_status(f"{task_status}")
        
        if request.form.get('SELECT_POSE') == "select_pose":
            selected_post = request.form.get('post_select')
            selected_post = int(selected_post)
            update_status(f"Set robot pose to {ros_server.robot_poses[selected_post]}")
            task_status = ros_server.send_pose_command(selected_post)
    
    # if ngrok:
    #     page = 'v2_ngrok.html'
    # else:
    #     page = 'v2.html'

    page = 'v2_ngrok.html'
    return render_template(page, logs=log,
                           grasp=ros_server.curr_n_grasps, 
                           pose=ros_server.robot_poses,
                           joint=ros_server.current_joint)

def update_status(txt="", space=False):
    global log
    
    if space:
        sp = "-"*5
        log += f"{sp}</br>"
    else:
        dt = datetime.datetime.now().strftime('%H:%M:%S')
        log += f"{dt} : {txt}</br>"
    
    return log


@app.route('/detected/<part>')
def detected(part):
    return Response(det(part),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def det(part):
    if ros_server.results is None:
        im = np.zeros((480,480,3), np.uint8)
    else:
        im = ros_server.results[part]
        
    frame = cv2.imencode('.jpg', im)[1].tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen(feed):
    while True:
        # Capture frame-by-frame
        frame = ros_server.get_frame(feed)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed/<feed>')
def video_feed(feed):
    return Response(gen(feed),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def startWeb():
    if debug_bool:
        app.run(host='0.0.0.0', debug=True)
        # app.run()
    else:
        app.run(host='0.0.0.0',  threaded=True)
    

debug_bool = True
# ngrok = True

ros_server = ROSBackend(debug_bool)

if __name__ == '__main__':
    print("Starting Server...")
    update_status("System Starting...")
    if debug_bool:
        startWeb()
    else:
        x = threading.Thread(target=startWeb)
        x.start()
        rospy.init_node('Simple_node_camera',anonymous=True)
        rospy.spin()