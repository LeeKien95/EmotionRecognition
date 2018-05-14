import asyncore
import socket
import ProcessLandmark as pl
import ProcessImage as pi
import simplejson as json
from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
import atexit
import cf_deployment_tracker
import os
import json
import sys


# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

deploy_ip = sys.argv[-1]

db_name = 'mydb'
client = None
db = None

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)


port = int(os.getenv('PORT', 8089))

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            # print(data["landmark_perk"])
            # print(data["landmark_neutral"])
            # print(data["image_data"])
            prediction = pl.getEmotionPredict(data["landmark_neutral"], data["landmark_perk"])
            subject_names = pi.getSubjectPredict(data["image_data"])
            prediction['subject_names'] = subject_names
            #print(prediction)
        except ValueError:
            return jsonify("Input Error.")

        return jsonify(prediction)


@app.route("/")
def hello():
    return "Hello World!"

@atexit.register
def shutdown():
    if client:
        client.disconnect()


if __name__ == '__main__':
    app.run(host=deploy_ip, port=port, debug=True)