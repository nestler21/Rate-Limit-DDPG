from flask import Flask, Response, request
from datetime import datetime, timedelta


# Simple Flask Server with an implemented Rate-Limit running on localhost

request_list = {}
window = timedelta(seconds=2) # window size in seconds
allowed_requests = 10 # number of allowed requests per time window, must be greater than 0

def check_rate_limit(user):
    if user not in request_list.keys():         # create user if not exists
        request_list[user] = [datetime.now()]
        return True
    
    request_list[user] = [timestamp for timestamp in request_list[user] if datetime.now() - timestamp < window] # update timestamp, remove outdated timestamps
    request_list[user].append(datetime.now())   # add current request
    if len(request_list[user]) <= allowed_requests:
        return True
    else:
        print(request_list)
        return False

app = Flask(__name__)

@app.route('/rate-limit')
def standard():
    user = request.remote_addr
    if check_rate_limit(user):
        return Response("200", status=200, mimetype='application/json')
    else:
        return Response("429", status=429, mimetype='application/json')

app.run()