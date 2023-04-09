from core import utils
import pickle
import os
import time
import re
import requests
import json
from datetime import datetime

from email import message
from dateutil.parser import parse
from email.utils import format_datetime
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

from email.utils import make_msgid


from core import connectors

class TokenExpired(Exception):
    pass

class WrongCredentials(Exception):
    pass

class ConnectionError(Exception):
    pass


class Reply(connectors.Content):
    def to_email(self):
        msg = message.EmailMessage()
        msg["Subject"] = "New reply"
        msg["From"] = self.username + "@instagram.com"
        msg["Message-ID"] = "<%s@instagram.com>" % self.id
        msg["Date"] = format_datetime(parse(self.timestamp))
        msg["In-Reply-To"] = "<%s@instagram.com>" % self.parent_id
        msg["References"] =  "<%s@instagram.com>" % self.media["id"] + " " + msg["In-Reply-To"]
        msg["X-Mailer"] = "Virtual Secretary"

        # Basic plain-text content for safety
        msg.set_content(self.text)

        # Embed the media - Only images for now.
        media = ""

        # Compose the body, use HTML for formatting
        html = """\
        <html style="background-color: #38383d">
            <body style="width: auto; max-width:640px; background-color: #f2f2f2; padding: 40px; color: black; margin: 0 auto;">
                <p>%s</p>
            </body>
        </html>
        """ % (self.text)
        msg.add_alternative(html, subtype='html')

        return msg

    def __init__(self, comment, server, content):
        self.server = server

        # Reply parent is a comment
        self.parent = comment

        # Add itself on top of the server objects list
        self.server.objects.append(self)

        # Content is a dict-like JSON reading, use that to lazily map keys to class properties
        self.__dict__.update(content)


class Comment(connectors.Content):
    def to_email(self):
        msg = message.EmailMessage()
        msg["Subject"] = "New comment"
        msg["From"] = self.username + "@instagram.com"
        msg["Message-ID"] = "<%s@instagram.com>" % self.id
        msg["Date"] = format_datetime(parse(self.timestamp))
        msg["In-Reply-To"] = "<%s@instagram.com>" % self.media["id"]
        msg["References"] = "<%s@instagram.com>" % self.media["id"]
        msg["X-Mailer"] = "Virtual Secretary"

        # Basic plain-text content for safety
        msg.set_content(self.text)

        # Embed the media - Only images for now.
        media = ""

        # Compose the body, use HTML for formatting
        html = """\
        <html style="background-color: #38383d">
            <body style="width: auto; max-width:640px; background-color: #f2f2f2; padding: 40px; color: black; margin: 0 auto;">
                <p>%s</p>
            </body>
        </html>
        """ % (self.text)
        msg.add_alternative(html, subtype='html')

        return msg

    def __init__(self, media, server, content):
        self.server = server
        self.objects = []

        # Comment parent is a media
        self.parent = media

        # Content is a dict-like JSON reading, use that to lazily map keys to class properties
        self.__dict__.update(content)

        # Add itself on top of the server objects list
        self.server.objects.append(self)

        # Comment children objects are replies
        if "replies" in content:
            [Reply(self, self.server, response) for response in self.replies["data"]]


class Media(connectors.Content):
    def to_email(self):
        msg = message.EmailMessage()
        msg["Subject"] = "New post : " + self.media_product_type.lower() + " " + self.media_type.lower()
        msg["From"] = self.username + "@instagram.com"
        msg["Message-ID"] = "<%s@instagram.com>" % self.id
        msg["Date"] = format_datetime(parse(self.timestamp))
        msg["X-Mailer"] = "Virtual Secretary"

        # Basic plain-text content for safety
        msg.set_content(self.caption + "\r\n" + self.permalink)

        # Embed the media - Only images for now.
        media = ""

        if self.media_type == "IMAGE" or self.media_type == "CAROUSEL_ALBUM":
            result = requests.get(self.media_url, stream=True)
            media = "<p><img src='%s' style='width: 100%%; height: auto;'/></p>" % self.media_url

        # Compose the body, use HTML for formatting
        html = """\
        <html style="background-color: #38383d">
            <body style="width: auto; max-width:640px; background-color: #f2f2f2; padding: 40px; color: black; margin: 0 auto;">
                %s
                <p>%s</p>
                <p><a href='%s'>Original post</a></p>
            </body>
        </html>
        """ % (media, self.caption, self.permalink)
        msg.add_alternative(html, subtype='html')

        # Attach the media as a backup if we were able to download it
        if result and result.status_code == 200:
            msg.add_attachment(result.content, maintype="image", subtype="jpg", filename="image.jpg")

        return msg

    def __init__(self, server, content):
        self.server = server
        self.parent = None

        # Content is a dict-like JSON reading, use that to lazily map keys to class properties
        self.__dict__.update(content)

        # Media children objects are comments
        if "comments" in content:
            [Comment(self, self.server, response) for response in self.comments["data"]]


class Message(connectors.Content):
    def __init__(self, parent, server, content):
        self.server = server

        # Message parent is a thread
        self.parent = parent

        # Content is a dict-like JSON reading, use that to lazily map keys to class properties
        self.__dict__.update(content)


class Thread(connectors.Content):
    def __init__(self, server, content):
        self.server = server

        # Content is a dict-like JSON reading, use that to lazily map keys to class properties
        self.__dict__.update(content)

        # Get messages for this thread
        result = requests.get("https://graph.facebook.com/v15.0/%s/?fields=messages{from,to,message,stories,stickers,attachments,shares}&access_token=%s" % (self.id, self.server.page_token))
        data = json.loads(result.text)

        # Thread children are messages
        self.objects = []
        [self.objects.append(Message(self, self.server, response)) for response in data["messages"]["data"]]


class Server(connectors.Server[Comment]):

    def get_objects(self):
        self.objects = []

        # Get media from the timeline
        result = requests.get("https://graph.facebook.com/v15.0/%s/media?fields=id,caption,media_product_type,media_url,permalink,thumbnail_url,username,timestamp,media_type,children,comments{id,text,username,timestamp,parent_id,media,replies{id,username,text,timestamp,parent_id,media}}&limit=50&access_token=%s" % (self.user, self.password))
        data = json.loads(result.text)
        [self.objects.append(Media(self, response)) for response in data["data"]]

        # Get DM threads
        self.threads = []
        #result = requests.get("https://graph.facebook.com/v15.0/%s/conversations?platform=instagram&fields=updated_time,participants&access_token=%s" % (self.business, self.page_token))
        #data = json.loads(result.text)
        #[self.threads.append(Thread(self, response)) for response in data["data"]]


    def __init_log_dict(self, log: dict):
        # Make sure the top-level dict keys are inited
        # for our current server/username pair
        if self.user not in log:
            log[self.user] = {}


    def __update_log_dict(self, id, log: dict, field: str, enable_logging: bool, action_run=True):
        if(enable_logging and action_run):
            try:
                # Update existing log entry for the current uid
                log[self.user][id][field] += 1
            except:
                # Create a new log entry for the current uid
                log[self.user][id] = {field: 1}

        if(not action_run):
            try:
                # Update existing log entry for the current uid
                log[self.user][id][field] += 0
            except:
                # Create a new log entry for the current uid
                log[self.user][id] = {field: 0}


    def run_filters(self, filter, action, runs=1):
        if len(self.objects) == 0:
            return

        # Define the log file as an hidden file inheriting the filter filename
        filtername = self.calling_file()
        directory = os.path.dirname(filtername)
        basename = os.path.basename(filtername)
        filter_logfile = os.path.join(directory, "." + basename + ".log")

        # Init a brand-new log dict
        log = { self.user:  # username
                {
                   # id : { PROPERTIES }
                }
              }

        # Enable logging and runs limitations if required
        enable_logging = (runs != -1) and isinstance(runs, int)

        # Open the logfile if any
        if os.path.exists(filter_logfile):
            with open(filter_logfile, "rb") as f:
                log = dict(pickle.load(f))
        else:
            pass

        self.__init_log_dict(log)

        ts = time.time()

        for elem in self.objects:
            # Disable the filters if the number of allowed runs is already exceeded
            if enable_logging and elem.id in log[self.user]:
                # We have a log entry for this hash.
                #print("%s found in DB" % email.hash)
                try:
                    filter_on = (log[self.user][elem.id]["processed"] < runs)
                except:
                    # We have a log entry but it's invalid. Reset it.
                    log[self.user][elem.id]["processed"] = 0
                    filter_on = True
            else:
                # We don't have a log entry for this hash or we don't limit runs
                #print("%s not found in DB" % email.hash)
                filter_on = True

            if filter_on and filter:
                try:
                    filter_on = filter(elem)
                except Exception as e:
                    print(e)

            if not isinstance(filter_on, bool):
                    filter_on = False
                    raise TypeError("The filter does not return a boolean, the behaviour is ambiguous.")

            # Run the action
            if filter_on and action:
                # The action should update self.std_out internally. If not, init here as a success.
                # Success and errors matter only for email write operations
                success = False
                try:
                    action(elem)

                    # Log success
                    # print("Filter application successful on", email["Subject"], "from", email["From"], "to", email["To"], "on", email["Date"])
                    self.__update_log_dict(elem.id, log, "processed", enable_logging)
                    success = True

                except Exception as e:
                    print(e)

                if not success:
                    # Log error
                    #print("Filter application failed on", email["Subject"], "from", email["From"], "to", email["To"], "on", email["Date"])
                    self.__update_log_dict(elem.id, log, "errored", enable_logging)

            else:
                # No action run but we still store email ID in DB
                self.__update_log_dict(elem.id, log, "processed", enable_logging, action_run=False)

        # Dump the log dict to a byte file for efficiency
        with open(filter_logfile, "wb") as f:
            pickle.dump(log, f)
            # print(log)
            #print("%s written" % filter_logfile)



    def check_token(self):
        # oAuth token is is defined and valid if it expires in the future, check timestamps
        if datetime.now() < datetime.fromtimestamp(self.token_access_expiration):
            # Send a test request to Facebook for user ID
            result = requests.get("https://graph.facebook.com/me?access_token=%s&fields=id" % self.password)

            if result.status_code == 200:
                # Return code 200 means HTTP requests is successful
                # Check if we have an ID in the response
                if "id" not in json.loads(result.text):
                    raise WrongCredentials
                else:
                    # Get the page token for the selected business ID, mandatory to access direct messages.
                    # NB : the user token is valid only to access comments
                    result = requests.get("https://graph.facebook.com/%s?fields=access_token&access_token=%s" % (self.business, self.password))
                    data = json.loads(result.text)

                    if "access_token" in data:
                        self.page_token = data["access_token"]
                    else:
                        raise WrongCredentials

            else:
                raise ConnectionError

        else:
            raise TokenExpired


    def init_connection(self, params: dict):
        # High-level method to login to a server in one shot
        # See https://sabre.io/dav/building-a-carddav-client/
        self.password = params["access_token"]
        self.server = "https://graph.facebook.com"
        self.user = params["instagram_business_account"]
        self.business = params["business_id"]
        self.token_access_expiration = float(params["data_access_expiration_time"])

        try:
            self.check_token()
            self.connection_inited = True
            self.get_objects()
        except TokenExpired:
            print("The authentification token is expired, you need to get a new one by executing the assistant `python src/facebook-login.py`")
        except:
            print("Can't connect and/or login to Facebook")



    def close_connection(self):
        # High-level method to logout from a server
        pass
