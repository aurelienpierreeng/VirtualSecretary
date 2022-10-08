import imaplib
import utils
import pickle
import os
import time
from protocols import imap_object

import connectors

class Server(connectors.Server[imap_object.EMail], imaplib.IMAP4_SSL):
    def get_imap_folders(self):
        # List inbox subfolders as plain text, to be reused by filter definitions

        mail_list = self.list()
        self.folders = []

        if(mail_list[0] == "OK"):
            for elem in mail_list[1]:
                entry = elem.decode().split('"."')
                flags = entry[0].strip("' ")
                folder = entry[1].strip("' ")

                if "\\Archive" in flags:
                    self.archive = folder
                if "\\Sent" in flags:
                    self.sent = folder
                if "\\Trash" in flags:
                    self.trash = folder
                if "\\Junk" in flags:
                    self.junk = folder

                self.folders.append(folder)

            self.logfile.write("%s : Found %i inbox subfolders : %s\n" % (
                utils.now(), len(self.folders), ", ".join(self.folders)))
        else:
            self.logfile.write(
                "%s : Impossible to get the list of inbox folders\n" % (utils.now()))

        self.std_out = mail_list

    def get_objects(self, mailbox: str, n_messages=-1):
        # List the n-th last emails in mailbox
        if not self.connection_inited:
            print("We do not have an active IMAP connection. Ensure your IMAP credentials are defined in `settings.ini`")
            return

        # If no explicit number of messages is passed, use the one from `settings.ini`
        if n_messages == -1:
            n_messages = self.n_messages

        if mailbox in self.folders:
            status, messages = self.select(mailbox)
            num_messages = int(messages[0])
            self.logfile.write("%s : Reached mailbox %s : %i emails found, loading only the first %i\n" % (
                utils.now(), mailbox, num_messages, n_messages))

            self.objects = []
            messages_queue = []

            # Network loop
            ts = time.time()

            try:
                # build a coma-separated list of IDs from start to end
                ids = range(max(num_messages - n_messages + 1, 1), num_messages + 1)
                ids = [str(x) for x in ids]
                ids = ",".join(ids)
                res, messages_queue = self.fetch(ids, "(FLAGS BODY.PEEK[] UID)")
            except:
                print(
                "Could not get some emails, they may have been deleted on server by another application in the meantime.")

            # When fetching emails in bulk, a weird "41" gets inserted between each record
            # so we need to keep one every next row.
            cleaned_queue = []
            i = 0
            for message in messages_queue:
                if i % 2 == 0:
                    cleaned_queue.append(message)
                i += 1

            print("  - IMAP\ttook %.3f s\tto query\t%i emails from %s" %
                  (time.time() - ts, len(cleaned_queue), mailbox))

            # Process loop
            ts = time.time()
            self.objects = [imap_object.EMail(msg, self) for msg in cleaned_queue]
            print("  - Parsing\ttook %.3f s\tto parse\t%i emails" %
                  (time.time() - ts, len(self.objects)))

            self.std_out = [status, messages]

        else:
            self.logfile.write("%s : Impossible to get the mailbox %s : no such folder on server\n" % (
                utils.now(), mailbox))

    def __init_log_dict(self, log: dict):
        # Make sure the top-level dict keys are inited
        # for our current server/username pair
        if self.server not in log:
            log[self.server] = {self.user: {}}

        if self.user not in log[self.server]:
            log[self.server][self.user] = {}

    def __update_log_dict(self, email: imap_object.EMail, log: dict, field: str, enable_logging: bool, action_run=True):
        if(enable_logging and action_run):
            try:
                # Update existing log entry for the current uid
                log[self.server][self.user][email.hash][field] += 1
            except:
                # Create a new log entry for the current uid
                log[self.server][self.user][email.hash] = {field: 1}

        if(not action_run):
            try:
                # Update existing log entry for the current uid
                log[self.server][self.user][email.hash][field] += 0
            except:
                # Create a new log entry for the current uid
                log[self.server][self.user][email.hash] = {field: 0}

    def run_filters(self, filter, action, runs=1):
        # Run the function `filter` and execute the function `action` if the filtering condition is met
        # * `filter` needs to return a boolean encoding the success of the filter.
        # * `action` needs to return a list where the [0] element contains "OK" if the operation succeeded,
        #    and the [1] element is user-defined stuff.
        # * `filter` and `action` take an `mailserver.EMail` instance as input
        # * `runs` defines how many times the emails need to be processed. -1 means no limit.
        # * `filter_file` is the full path to the filter Python script

        # Define the log file as an hidden file inheriting the filter filename
        filtername = self.calling_file()
        directory = os.path.dirname(filtername)
        basename = os.path.basename(filtername)
        filter_logfile = os.path.join(directory, "." + basename + ".log")

        # Init a brand-new log dict
        log = {self.server:  # server
               {self.user:  # username
                {
                    # email.hash : { PROPERTIES }
                }
                }
               }

        # Enable logging and runs limitations if required
        enable_logging = (runs != -1) and isinstance(runs, int)

        # Open the logfile if any
        if enable_logging and os.path.exists(filter_logfile):
            with open(filter_logfile, "rb") as f:
                log = dict(pickle.load(f))
                #print("DB found : %s" % f)
                # print(log)
        else:
            #print("%s not found" % log)
            pass

        self.__init_log_dict(log)

        ts = time.time()

        for email in self.objects:
            # Disable the filters if the number of allowed runs is already exceeded
            if enable_logging and email.hash in log[self.server][self.user]:
                # We have a log entry for this hash.
                #print("%s found in DB" % email.hash)
                filter_on = (log[self.server][self.user]
                             [email.hash]["processed"] < runs)
            else:
                # We don't have a log entry for this hash or we don't limit runs
                #print("%s not found in DB" % email.hash)
                filter_on = True

            # Run the actual filter
            if filter_on and filter:
                if True:
                    # User wrote good filters.
                    self.std_out = ["", ]
                    filter_on = filter(email)
                    # print(filter_on)

                    if not isinstance(filter_on, bool):
                        filter_on = False
                        print(
                            "The filter does not return a boolean, the behaviour is ambiguous. Filtering is canceled.")
                        raise TypeError(
                            "The filter does not return a boolean, the behaviour is ambiguous. Filtering is canceled.")
                else:
                    # User tried to filter non-existing fields or messed-up somewhere.
                    filter_on = False

            # Run the action
            if filter_on:
                try:
                    # The action should update self.std_out internally. If not, init here as a success.
                    # Success and errors matter only for email write operations
                    self.std_out = ["", ]
                    action(email)

                    if self.std_out[0] == "OK":
                        # Log success
                        print("Filter application successful on",
                              email["Subject"], "from", email["From"])
                        self.__update_log_dict(
                            email, log, "processed", enable_logging)
                    else:
                        # Log error
                        print("Filter application failed on",
                              email["Subject"], "from", email["From"])
                        self.__update_log_dict(
                            email, log, "errored", enable_logging)

                except:
                    # Log error
                    print("Filter application failed on",
                          email["Subject"], "from", email["From"])
                    self.__update_log_dict(
                        email, log, "errored", enable_logging)
            else:
                # No action run but we still store email ID in DB
                self.__update_log_dict(
                    email, log, "processed", enable_logging, action_run=False)

        # Actually delete with IMAP the emails marked with the tag `\DELETED`
        # We only need to run it once per email loop/mailbox.
        self.expunge()

        # Close the current mailbox.
        # We need it here in case a filter starts more than one loop over different mailboxes
        self.close()

        print("  - Filtering\ttook %.3f s\tto filter\t%i emails" %
              (time.time() - ts, len(self.objects)))

        # Dump the log dict to a byte file for efficiency
        with open(filter_logfile, "wb") as f:
            pickle.dump(log, f)
            # print(log)
            #print("%s written" % filter_logfile)

    def init_connection(self, params: dict):
        # High-level method to login to a server in one shot
        self.n_messages = int(params["entries"])
        self.server = params["server"]
        self.user = params["user"]

        # Init the SSL connection to the server
        self.logfile.write("%s : Trying to login to %s with username %s\n" % (
            utils.now(), self.server, self.user))
        try:
            imaplib.IMAP4_SSL.__init__(self, host=self.server)
        except:
            print(
                "We can't reach the server %s. Check your network connection." % self.server)

        try:
            self.std_out = self.login(self.user, params["password"])
            self.logfile.write("%s : Connection to %s : %s\n" %
                               (utils.now(), self.server, self.std_out[0]))
        except:
            print("We can't login to %s with username %s. Check your credentials" % (
                self.server, self.user))

        self.get_imap_folders()

        self.connection_inited = True

    def close_connection(self):
        # High-level method to logout from a server
        self.logout()
        self.connection_inited = False
