from flask import Flask, render_template, request, redirect, jsonify, url_for, session, abort
import requests
from authlib.integrations.flask_client import OAuth
import os

app = Flask(__name__)

app.secret_key ='deepsecret'
oauth = OAuth(app)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/google')
def google():
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    CONF_URL = 'https://accounts.google.com/.well-known/openid-configuration'
    oauth.register(
        name='google',
        client_id= GOOGLE_CLIENT_ID,
        client_secret= GOOGLE_CLIENT_SECRET,
        server_metadata_url= CONF_URL,
        client_kwargs={
            'scope': 'openid email profile'
        }
    )
    # Redirect to google_auth function
    redirect_uri = url_for('google_auth', _external=True)
    # print(f"Redirect URI: {redirect_uri}")  
    return oauth.google.authorize_redirect(redirect_uri)
 
@app.route('/google/auth')
def google_auth():
    token = oauth.google.authorize_access_token()
    nonce = request.args.get('nonce')  # Retrieve the nonce from the request
    user = oauth.google.parse_id_token(token, nonce=nonce)  
    print("Google User:", user)
    return redirect('/index')

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
