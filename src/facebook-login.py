from datetime import datetime
from oauthlib.oauth2 import MobileApplicationClient
from requests_oauthlib import OAuth2Session

from PySide6.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QMainWindow, QRadioButton, QFileDialog
from PySide6.QtCore import QUrl, QSize
from PySide6.QtWebEngineWidgets import QWebEngineView

import configparser
import os
import requests
import json

class FbConnector():
  # Oauth params
  AUTH_URL = "https://www.facebook.com/v15.0/dialog/oauth"
  REDIRECT_URI = "https://www.facebook.com/connect/login_success.html"
  RETURN_TYPE = "token"
  SCOPE = ["email",
           "instagram_basic", "instagram_manage_comments", "instagram_manage_messages", "instagram_content_publish",
           "pages_show_list", "pages_manage_metadata", "pages_messaging"]


  def build_oauth_request(self):
    self.oauth = OAuth2Session(client = MobileApplicationClient(client_id = self.client_id),
                               redirect_uri = self.REDIRECT_URI, scope = self.SCOPE)
    self.authorization_url, self.state = self.oauth.authorization_url(self.AUTH_URL)
    print(self.authorization_url)


  def extract_token(self, url):
    # GUI callback on "url changed" event in webbrowser view
    print(url.toString())
    try:
      token = self.oauth.token_from_fragment(url.toString())
      if token["state"] == self.state:
        # Request didn't got tampered with during transport
        # Write our config file
        self.config["instagram"] = token
        self.config["instagram"]["client_id"] = self.client_id

        # Close the web view and save the config
        self.view.close()
        self.write_config()
        self.reset_center_widget()
        self.get_accounts()
    except:
      pass


  def check_token(self):
    # Check if we have n `.ini` file containing a valid Instagram oAuth token.
    is_valid = False

    if "instagram" in self.config:
      if "access_token" in self.config["instagram"] and "data_access_expiration_time" in self.config["instagram"]:
        # oAuth token is is defined and valid if it expires in the future, check timestamps
        if datetime.now() < datetime.fromtimestamp(float(self.config["instagram"]["data_access_expiration_time"])):
          # Send a test request to Facebook for user ID
          result = requests.get("https://graph.facebook.com/me?access_token=%s&fields=id" % self.config["instagram"]["access_token"])

          if result.status_code == 200:
            # Return code 200 means HTTP requests is successful
            data = json.loads(result.text)

            # Check if we have an ID in the response
            is_valid = "id" in data

        print("token found and valid until", datetime.fromtimestamp(float(self.config["instagram"]["data_access_expiration_time"])))

    return is_valid


  def reset_center_widget(self):
    # Bruteforce way of erasing all content in the main window
    self.HBox = QVBoxLayout()
    self.HBox.setSpacing(2)
    self.center = QWidget()
    self.center.setLayout(self.HBox)
    self.window.setCentralWidget(self.center)


  def fb_login(self):
    # Load a web view with FB login in it
    self.reset_center_widget()
    self.view = QWebEngineView()
    self.view.load(QUrl(self.authorization_url))
    self.view.setFixedSize(QSize(1200, 800))
    self.view.urlChanged.connect(self.extract_token)
    self.HBox.addWidget(self.view)


  def get_oauth(self):
    self.client_id = self.client_id_widget.text()
    self.build_oauth_request()
    self.fb_login()


  def input_client(self):
      # Set the client ID
    label = QLabel("Input your Facebook app ID : <a href=\"https://developers.facebook.com/apps\">(Where to find it ?)</a>")
    label.setOpenExternalLinks(True)
    self.client_id_widget = QLineEdit()

    # Init it from config file if available
    if "instagram" in self.config and "client_id" in self.config["instagram"]:
      self.client_id_widget.setText(self.config["instagram"]["client_id"])

    # Set button and GUI stuff
    self.button = QPushButton("Get authentication token")
    self.button.clicked.connect(self.get_oauth)

    self.reset_center_widget()
    self.HBox.addWidget(label)
    self.HBox.addWidget(self.client_id_widget)
    self.HBox.addWidget(self.button)


  def set_account(self, radio_button, id):
    self.business_id = id
    self.reset_center_widget()
    result = requests.get("https://graph.facebook.com/v15.0/%s?fields=instagram_business_account&access_token=%s" \
      % (self.business_id, self.config["instagram"]["access_token"]))

    if result.status_code == 200:
      data = json.loads(result.text)
      print(data)
      self.config["instagram"]["business_id"] = self.business_id

      if "instagram_business_account" in data:
        self.config["instagram"]["instagram_business_account"] = data["instagram_business_account"]["id"]
        self.reset_center_widget()
        self.write_config()

        result = requests.get("https://graph.facebook.com/v15.0/%s?fields=name,followers_count,username&access_token=%s" \
                                  % (self.config["instagram"]["instagram_business_account"], self.config["instagram"]["access_token"]))

        if result.status_code == 200:
          data = json.loads(result.text)
          confirmation = QLabel("We found the Instagram business account <strong>%s</strong> (@%s), having %s followers." \
                                  % (data["name"], data["username"], data["followers_count"]))
          self.HBox.addWidget(confirmation)

          info = QLabel("Your credentials have been successfully saved to %s, you can now close this window." % self.config_file_path)
          self.HBox.addWidget(info)
      else:
        self.reset_center_widget()
        info = QLabel("It seems that you have no Instagram account linked to this business account.")
        self.HBox.addWidget(info)

    else:
      self.reset_center_widget()
      info = QLabel("We couldn't reach Facebook server, check your internet connection and restart the assistant.")
      self.HBox.addWidget(info)

  def get_accounts(self):
    self.reset_center_widget()
    result = requests.get("https://graph.facebook.com/v15.0/me/accounts?access_token=%s" % self.config["instagram"]["access_token"])

    if result.status_code == 200:
      data = json.loads(result.text)
      print(data["data"])

      if "data" in data:
        info = QLabel("Your authentification token is valid until %s" % datetime.fromtimestamp(float(self.config["instagram"]["data_access_expiration_time"])))
        self.HBox.addWidget(info)

        label = QLabel("Choose the one business account you want to configure :")
        self.HBox.addWidget(label)

        for elem in data["data"]:
          if "MESSAGING" in elem["tasks"] and "MODERATE" in elem["tasks"] and "CREATE_CONTENT" in elem["tasks"]:
            button = QRadioButton(elem["name"] + ", " + elem["category"])
            button.setAccessibleName(elem["id"])
            button.toggled.connect(lambda: self.set_account(button, elem["id"]))
            self.HBox.addWidget(button)

      else:
        self.reset_center_widget()
        info = QLabel("It seems that you have no business account/Facebook page linked to this Facebook account.")
        self.HBox.addWidget(info)
    else:
      self.reset_center_widget()
      info = QLabel("We couldn't reach Facebook server, check your internet connection and restart the assistant.")
      self.HBox.addWidget(info)


  def read_config(self):
    if os.path.exists(self.config_file_path):
      self.config.read(self.config_file_path)
      print("Instagram config file successfully read from %s" % self.config_file_path)


  def write_config(self):
    with open(self.config_file_path, 'w') as configfile:
      self.config.write(configfile)
      print("Instagram token successfully exported to %s" % self.config_file_path)


  def __init__(self, config_file_path="instagram.ini"):
    # Fetch previous config if any
    self.config_file_path = config_file_path
    self.config = configparser.ConfigParser()
    self.read_config()

    # Open an assistant window
    app = QApplication()
    self.window = QMainWindow()
    self.window.setWindowTitle("Virtual Secretary : connect to your Facebook account")
    self.window.show()

    # Start the IG dance
    if self.check_token():
      self.get_accounts()
    else:
      self.input_client()

    app.exec()


if __name__ == '__main__':
  FbConnector()
