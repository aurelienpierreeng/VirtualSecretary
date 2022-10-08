import smtplib

import connectors

from email import message
from email.utils import formatdate, make_msgid

class Server(connectors.Server[connectors.Content], smtplib.SMTP_SSL):

    def get_objects(self):
        pass

    def run_filters(self, filter, action, runs=1):
        pass

    def write_message(self, subject: str, to: str, content: str):
        # Prepare an email with the usuals set
        self.msg = message.EmailMessage()
        self.msg['Subject'] = subject
        self.msg['From'] = self.user
        self.msg['To'] = to
        self.msg["Message-ID"] = make_msgid(domain=self.server)
        self.msg["Date"] = formatdate()
        self.msg.set_content(content)

    def send_message(self):
        super().send_message(self.msg)

    def init_connection(self, params: dict):
        # High-level method to login to a server in one shot
        password = params["password"]
        self.server = params["server"]
        self.user = params["user"]

        smtplib.SMTP_SSL.__init__(self, self.server)
        self.ehlo(self.server)
        self.login(self.user, password)

        # Notify that we have a server with an active connection
        self.connection_inited = True


    def close_connection(self):
        # High-level method to logout from a server
        self.close()
