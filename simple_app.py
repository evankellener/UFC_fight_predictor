#!/usr/bin/env python3
"""
Simple Flask app for testing Heroku deployment
"""

from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <h1>UFC Fight Predictor - Test Deployment</h1>
    <p>✅ Flask app is working!</p>
    <p>✅ Heroku deployment successful!</p>
    <p>Current directory: {}</p>
    <p>Python version: {}</p>
    '''.format(os.getcwd(), os.sys.version)

@app.route('/test')
def test():
    return {'status': 'success', 'message': 'API is working!'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
