import smtplib
import imaplib

import connectors
import utils

from email import message
from email.utils import formatdate, make_msgid, parsedate_to_datetime

class Server(connectors.Server[connectors.Content], smtplib.SMTP_SSL):

    def get_objects(self):
        pass

    def run_filters(self, filter, action, runs=1):
        pass

    def write_message(self, subject: str, to: str, content: str, reply_to : message.EmailMessage = None):
        # Prepare an email with the usuals fields
        # If this is an answer, pass on the original EMail object in `reply_to`,
        # the references and Message-ID are handled automatcally

        self.msg = message.EmailMessage()
        self.msg["Subject"] = subject
        self.msg["From"] = self.user
        self.msg["To"] = to
        self.msg["Message-ID"] = make_msgid(domain=self.server)
        self.msg["Date"] = formatdate(localtime=True)
        self.msg["X-Mailer"] = "Virtual Secretary"

        # Pass on the ID of the email being replied to, and previous emails in the thread if any.
        # This is to support email threads.
        if reply_to:
            if "Message-ID" in reply_to.headers:
                self.msg["In-Reply-To"] = reply_to["Message-ID"]

                if "References" in reply_to.headers:
                    self.msg["References"] = reply_to["References"]

                if "References" in self.msg:
                    self.msg["References"] += " " + reply_to["Message-ID"]
                else:
                    self.msg["References"] = reply_to["Message-ID"]

        self.msg.set_content(content)


    def send_message(self, copy_to_sent=False):
        # If copy_to_sent=True, the message is added to the "Sent" IMAP folder
        # after being sent to receipient.
        # A valid IMAP server connection needs to be open.
        self.reinit_connection()
        super().send_message(self.msg)

        if copy_to_sent and "imap" in self.secretary.protocols and self.secretary.protocols["imap"].connection_inited:
            # Get the default "Sent" folder on this IMAP server
            sentbox = self.secretary.protocols["imap"].sent

            # Reformat the email date for IMAP
            date = parsedate_to_datetime(self.msg["Date"])
            date = imaplib.Time2Internaldate(date)

            try:
                # We need to refresh the connection in case we timed out and were logged out
                self.secretary.protocols["imap"].reinit_connection()
                result = self.secretary.protocols["imap"].append(sentbox, "(\\Seen Auto)", date, self.msg.as_bytes())

                if result[0] == "OK":
                    self.logfile.write("%s : Copied (UID %s) `%s` to `%s` sent on %s to %s\n" % (utils.now(),
                                                                                                self.msg["Message-ID"],
                                                                                                self.msg["Subject"],
                                                                                                self.msg["To"],
                                                                                                self.msg["Date"],
                                                                                                sentbox))
                else:
                    self.logfile.write("%s : Failed to copy (UID %s) `%s` to `%s` sent on %s to %s\n" % (utils.now(),
                                                                                                        self.msg["Message-ID"],
                                                                                                        self.msg["Subject"],
                                                                                                        self.msg["To"],
                                                                                                        self.msg["Date"],
                                                                                                        sentbox))
            except:
                self.logfile.write("%s : Failed to copy (UID %s) `%s` to `%s` sent on %s to %s\n" % (utils.now(),
                                                                                                    self.msg["Message-ID"],
                                                                                                    self.msg["Subject"],
                                                                                                    self.msg["To"],
                                                                                                    self.msg["Date"],
                                                                                                    sentbox))


    def reinit_connection(self):
        # Deal with timeouts
        try:
            smtplib.SMTP_SSL.__init__(self, self.server, port=self.port)
        except:
            print("[SMTP] We can't reach the server %s. Check your network connection." % self.server)

        try:
            self.ehlo(self.server)
            self.std_out = self.login(self.user, self.password)
            logstring = "[SMTP] Connection to %s : %s" % (self.server, "OK" if self.std_out[0] == 235 else self.std_out[0])
            self.logfile.write("%s : %s\n" % (utils.now(), logstring))
            print(logstring)
        except:
            print("We can't login to %s with username %s. Check your credentials" % (
                self.server, self.user))


    def init_connection(self, params: dict):
        # High-level method to login to a server in one shot
        self.password = params["password"]
        self.server = params["server"]
        self.user = params["user"]
        self.port = int(params["port"]) if "port" in params else 465

        # Notify that we have a server with an active connection
        logstring = "[SMTP] Trying to login to %s with username %s" % (self.server, self.user)
        self.logfile.write("%s : %s\n" % (utils.now(), logstring))
        print(logstring)
        self.reinit_connection()
        self.connection_inited = True


    def close_connection(self):
        # High-level method to logout from a server
        self.close()
