from __future__ import annotations

import imaplib
import email
import utils
import pickle
import os
import time
import re
from protocols import imap_object

from email.utils import formatdate, make_msgid, parsedate_to_datetime

import connectors

class Server(connectors.Server[imap_object.EMail], imaplib.IMAP4_SSL):
    """IMAP server connector using mandatory SSL. Non-SSL connection is not implemented on purpose, because the Internet is a dangerous place and SSL only adds a little bit of safety, but it's better than going out naked.

    This class inherits from the Python standard class [imaplib.IMAP4_SSL][], so all the method are available, although most of them are re-wrapped here for direct and higher-level data handling.
    """

    def build_subfolder_name(self, path: list) -> str:
        """Assemble a complete subfolder name using the separator of the server.

        Path should be the complete list of parent folders, e.g.
        `path = ["INBOX", "Money", "Taxes"]` will be assembled as
        `INBOX.Money.Taxes` or `INBOX/Money/Taxes`, depending on server's defaults.

        Then, replace the `INBOX` marker with the actual case-sensitive inbox name.
        This is to deal with Outlook/Office365 discrepancies in folders name.

        Arguments:
            path (list): the tree of parents folders

        Returns:
            path (str): IMAP-encoded UTF-8 path
        """
        return self.separator.join(path).replace("INBOX", self.inbox)


    def split_subfolder_path(self, folder: str) -> list[str]:
        """Find out what kind of separator is used on server for IMAP subfolder and split the parent folders into a list of folders.

        Most servers use dots, like `INBOX.Money.Taxes`, but Outlook/Office365 uses slashes, like `INBOX/Money/Taxes`.

        Arguments:
            folder (str): IMAP folder path

        Returns:
            tree (list): list of parent folders
        """

        if re.match(r".*?\..*", folder):
            # We found a dot-separated subfolder
            path = folder.split(".")
        elif re.match(r".*?\/.*", folder):
            # We found a slash-sparated subfolder
            path = folder.split("/")
        else:
            # We have a first-level folder. Make it a single-element list for uniform handling
            path = [folder]

        return path


    def encode_imap_folder(self, folder:str) -> bytes:
        """Ensure the subfolders are properly separated using the actual server separator (`.` or `/`) and encode the names in IMAP-custom UTF-7, taking care of non-latin characters and enquoting strings containing whitespaces. The result is ready for direct use in IMAP server commands.

        This function takes fully-formed IMAP mailbox folder pathes, like `INBOX.Money.Taxes` or `INBOX/Money/Taxes` and will replace the subfolder separators with the server separator. The main `INBOX` will also be replaced by the proper, case-sensitive, inbox name for the current server.

        Arguments:
            folder (str): IMAP folder as Python string

        Returns:
            folder (bytes): IMAP folder as IMAP-custom UTF-7 bytes.
        """

        # Remove spaces and quotes around folder name, if any
        folder_str = folder.strip("' \"")

        # Rebuild the new path string using the current server separator
        path = self.split_subfolder_path(folder_str)
        folder_str = self.build_subfolder_name(path)

        # Insert into quotes if apostrophe or spaces are found in name
        if " " in folder_str or "'" in folder_str:
            folder_str = "\"" + folder_str + "\""

        # Convert to bytes as UTF-7 with custom IMAP mapping for special characters
        return utils.imap_encode(folder_str)


    def get_imap_folders(self):
        """List all inbox subfolders as plain text, to be reused by filter definitions. Update the [Server.folders][protocols.imap_server.Server.folders] list."""

        mail_list = self.list()
        self.folders = []

        if(mail_list[0] == "OK"):
            for elem in mail_list[1]:
                # Subfolders can be separated by . or by / depending on servers, let's find out
                tokens = re.match(r"\((.*?)\) \"([.\/])\" (.*)", utils.imap_decode(elem)).groups()

                flags = tokens[0]
                self.separator = tokens[1]
                folder = tokens[2].strip("' \"")

                if "inbox" == folder.lower():
                    # IMAP RFC something says the base inbox should be called `INBOX` (case-sensitive)
                    # Outlook/Office365 calls it `Inbox`. Deal with that.
                    self.inbox = folder
                if "\\Archive" in flags:
                    self.archive = folder
                if "\\Sent" in flags:
                    self.sent = folder
                if "\\Trash" in flags:
                    self.trash = folder
                if "\\Junk" in flags:
                    self.junk = folder
                if "\\Drafts" in flags:
                    self.drafts = folder
                if "\\Flagged" in flags:
                    self.flagged = folder

                self.folders.append(folder)

            self.logfile.write("%s : Found %i inbox subfolders : %s\n" % (
                utils.now(), len(self.folders), ", ".join(self.folders)))
        else:
            self.logfile.write(
                "%s : Impossible to get the list of inbox folders\n" % (utils.now()))

        self.std_out = mail_list


    def get_email(self, uid: str, mailbox=None) -> imap_object.EMail:
        """Get an arbitrary email by its UID. If no mailbox is specified,
        use the one defined when we got objects in `self.set_objects()`.

        If a mailbox is specified, we select it temporarilly and we restore the original mailbox used to get objects. If no mailbox is selected, we use the previously-selected one, typically in [Server.get_objects][protocols.imap_server.Server.get_objects].

        Arguments:
            uid (str): the unique ID of the email in the mailbox. Be aware that this ID is unique only in the scope of one mailbox (aka IMAP (sub)folder) because it is defined as the positional order of reception of each email in the mailbox.

        Returns:
            message (imap_object.EMail): the email object.
        """
        if mailbox:
            self.select(mailbox)

        message = None
        result, msg = self.uid('FETCH', uid, '(UID FLAGS BODY.PEEK[])', None)

        if result == "OK" and msg[0]:
            message = imap_object.EMail(msg[0], self)

        if mailbox:
            self.select(mailbox)

        return message


    def get_objects(self, mailbox: str, n_messages=-1):
        """Get the n last emails in a mailbox.

        Arguments:
            mailbox (str): the full path of the mailbox. It will be sanitized for folder/subfolder separator and actual `INBOX` name internally.
            n_messages (int): number of messages to fetch, starting with the most recent. If `-1`, the preference set in `settings.ini` will be used. Any other value will set it temporarily.

        """
        if not self.connection_inited:
            print("We do not have an active IMAP connection. Ensure your IMAP credentials are defined in `settings.ini`")
            return

        # If no explicit number of messages is passed, use the one from `settings.ini`
        if n_messages == -1:
            n_messages = self.n_messages

        encoded_mailbox = self.build_subfolder_name(self.split_subfolder_path(mailbox))

        if encoded_mailbox not in self.folders:
            self.logfile.write("%s : Impossible to get the mailbox %s : no such folder on server\n" % (
                utils.now(), mailbox))
            return

        self.mailbox = encoded_mailbox
        self.objects = []
        messages_queue = []

        # Network loop
        ts = time.time()
        retries = 0
        has_something = True

        while not messages_queue and retries < 5 and has_something:
            # Retry for as long as we didn't fetch as many messages as we have on the server
            try:
                # Avoid getting logged out by time-outs
                self.reinit_connection()
                status, messages = self.select(self.mailbox)
                num_messages = int(messages[0])

                if (status == "OK" and num_messages == 0) or status == "NO":
                    # There is no email in selected folder or we don't have access permission, abort
                    print("  No email in this mailbox")
                    has_something = False
                    break

                self.logfile.write("%s : Reached mailbox %s : %i emails found, loading only the first %i\n" % (
                    utils.now(), mailbox, num_messages, n_messages))

                # build a coma-separated list of IDs from start to end
                ids = [str(x) for x in range(max(num_messages - n_messages + 1, 1), num_messages + 1)]
                ids = ",".join(ids)
                res, messages_queue = self.fetch(ids, "(UID FLAGS BODY.PEEK[])")

            except:
                retries += 1

                # Wait some seconds before next connection attempt.
                # Increase the timeout in case there is a threshold on server
                print("  Could not get the emails from %s, will retry in %i s." % (mailbox, retries))
                time.sleep(retries)

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

        # Append emails only if there is a message body saved as bytes.
        # no message body means the email got deleted on server while we fetched it.
        self.objects = [imap_object.EMail(msg, self) for msg in cleaned_queue if msg and len(msg) > 1 and msg[1] and isinstance(msg[1], bytes)]
        print("  - Parsing\ttook %.3f s\tto parse\t%i emails" % (time.time() - ts, len(self.objects)))

        self.std_out = [status, messages]



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
        """Run the function `filter` and execute the function `action` if the filtering condition is met

        Arguments:
            filter (callable): function performing checking arbitrary conditions on emails, returning `True` if the action should be performed. It will get an [EMail][protocols.imap_object.EMail] object as argument.
            action (callable): function performing the actual action. It will get an [EMail][protocols.imap_object.EMail] object as argument.
            runs (int): how many times a filter should run at most on each email. `-1` means no limit.

        """
        if not self.objects:
            # No queue
            return

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
        filter_on = True

        # Open the logfile if any
        if os.path.exists(filter_logfile):
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
                try:
                    filter_on = (log[self.server][self.user]
                                [email.hash]["processed"] < runs)
                except:
                    # We have a log entry but it's invalid. Reset it.
                    log[self.server][self.user][email.hash]["processed"] = 0
                    filter_on = True
            else:
                # We don't have a log entry for this hash or we don't limit runs
                #print("%s not found in DB" % email.hash)
                filter_on = True

            # Run the actual filter
            if filter_on and filter:
                # User wrote good filters.
                self.std_out = ["OK", ]

                try:
                    filter_on = filter(email)
                    # print("Filter successful on", email["Subject"], "from", email["From"], "to", email["To"], "on", email["Date"])
                except Exception as e:
                    print("Filter failed on", email["Subject"], "from", email["From"], "to", email["To"], "on", email["Date"])
                    print(e)
                    filter_on = False

                if not isinstance(filter_on, bool):
                    filter_on = False
                    raise TypeError("The filter does not return a boolean, the behaviour is ambiguous.")

            # Run the action
            if filter_on and action:
                # The action should update self.std_out internally. If not, init here as a success.
                # Success and errors matter only for email write operations
                self.std_out = ["OK", ]
                self.reinit_connection()

                success = False
                try:
                    # self.std_out =
                    action(email)
                    # TODO: make all actions return ["OK", etc.]

                    if self.std_out[0] == "OK":
                        # Log success
                        # print("Filter application successful on", email["Subject"], "from", email["From"], "to", email["To"], "on", email["Date"])
                        self.__update_log_dict(email, log, "processed", enable_logging)
                        success = True

                except Exception as e:
                    print("Action failed on", email["Subject"], "from", email["From"], "to", email["To"], "on", email["Date"])
                    print(e)

                if not success:
                    # Log error
                    print("Filter application failed on", email["Subject"], "from", email["From"], "to", email["To"], "on", email["Date"])
                    self.__update_log_dict( email, log, "errored", enable_logging)

            else:
                # No action run but we still store email ID in DB
                self.__update_log_dict(email, log, "processed", enable_logging, action_run=False)


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


    def count(self, method):
        # Count the number of messages matching a criterion defined in the EMail.method()
        # Criteria will typically the presence of some flags
        # TODO: test it ?
        number = 0
        for email in self.objects:
            func = getattr(email, method)
            if func():
                number += 1

        return number


    def init_connection(self, params: dict):
        """High-level method to login to a server in one shot. The parameters are passed by the [secretary.Secretary.load_connectors][] caller.

        Arguments:
            params (dict): the dictionnary of mailserver credentials, extracted from the `settings.ini` file by [configparser.ConfigParser][].

        Examples:
            Mandatory content of the `settings.ini` file to declare IMAP connection credentials:

            ```ini
            [imap]
                user = me@server.com
                password = xyz
                server = mail.server.com
                entries = 20
            ```
        """
        self.n_messages = int(params["entries"])
        self.server = params["server"]
        self.user = params["user"]
        self.password = params["password"] # TODO: hash that in RAM ?
        self.port = int(params["port"]) if "port" in params else 993

        # Init the SSL connection to the server
        logstring = "[IMAP] Trying to login to %s with username %s" % (self.server, self.user)
        self.logfile.write("%s : %s\n" % (utils.now(), logstring))
        print(logstring)
        self.reinit_connection()
        self.get_imap_folders()
        self.connection_inited = True


    def probe_connection(self):
        """Probe the connection to see if it's still open"""
        try:
            self.std_out = self.noop()
            return self.std_out[0] == "OK"
        except:
            return False


    def reinit_connection(self):
        """Restart the IMAP connection using previous parameters, to prevent deconnections from time-outs"""
        if self.probe_connection():
            # We still have an open connection, nothing to reinit
            return

        # No connection open, need to (re)start one
        try:
            imaplib.IMAP4_SSL.__init__(self, host=self.server, port=self.port)
        except:
            print("[IMAP] We can't reach the server %s. Check your network connection." % self.server)
            return

        try:
            self.std_out = self.login(self.user, self.password)
            logstring = "[IMAP] Connection to %s : %s" % (self.server, self.std_out[0])
            self.logfile.write("%s : %s\n" % (utils.now(), logstring))
            print(logstring)
        except:
            print("We can't login to %s with username %s. Check your credentials" % (
                self.server, self.user))
            return

        # Need to re-select the mailbox if any
        if self.mailbox:
            self.select(self.mailbox)


    def close_connection(self):
        """High-level method to logout from a server"""

        self.logout()
        self.connection_inited = False

    def create_folder(self, folder:str):
        """Create an IMAP (sub)folder recursively if needed (create the parent(s) if missing, then create the child).

        Calls [Server.create][protocols.imap_server.Server.create] at each recursivity level, so IMAP folder names are fully sanitized.
        """

        path = []
        for level in self.split_subfolder_path(folder):
            path.append(level)
            target = self.build_subfolder_name(path)

            if target not in self.folders:
                result = self.create(target)

                if result[0] == "OK":
                    print("Folder `%s` created\n" % target)
                    self.logfile.write("%s : Folder `%s` created\n" % (utils.now(), target))
                    self.subscribe(target)
                else:
                    print("Failed to create folder `%s`\n" % target)
                    self.logfile.write("%s : Failed to create folder `%s`\n" % (utils.now(), target))

        # Update the list of server folders
        self.get_imap_folders()

    ## Re-implement imaplib folder commands with our folder encoding on top
    def select(self, mailbox: str):
        return super().select(self.encode_imap_folder(mailbox))

    def subscribe(self, mailbox: str):
        return super().subscribe(self.encode_imap_folder(mailbox))

    def create(self, mailbox: str):
        """Create a new mailbox. This is a wrapper over [imaplib.IMAP4.create][] method, where `mailbox` is directly encoded to IMAP-custom UTF-7 format through [Server.encode_imap_folder][protocols.imap_server.Server.encode_imap_folder].

        The direct use of this function is discouraged, see [Server.create_folder][protocols.imap_server.Server.create_folder] for a nicer method. Namely, this will error if trying to create a subfolder of a non-existing parent folder, and does not update [Server.folders][protocols.imap_server.Server.folders].
        """
        return super().create(self.encode_imap_folder(mailbox))

    def append(self, mailbox: str, flags: str, email: email.message.EmailMessage) -> str:
        """Add an arbitrary email to the specified mailbox. The email doesn't need to have been actually sent, it can be generated programmatically or be a copy of a sent email.

        Arguments:
            mailbox (str): the name of the target mailbox (folder or subfolder).
            flags (str): IMAP flags, standard (like `'(\\Seen)'` to mark as read, or `'(\\Flagged)'` to mark as important) or custom (can be any string not starting with `\`).
            email (email.message.EmailMessage): a proper `EmailMessage` object with initialized content, ready to send.

        Returns:
            status (str)
        """
        self.reinit_connection()
        self.create_folder(mailbox)

        # Reformat RFC 822 date to IMAP stuff
        date = parsedate_to_datetime(email["Date"])
        date = imaplib.Time2Internaldate(date)
        return super().append(self.encode_imap_folder(mailbox), flags, date, email.as_bytes())

    def __init__(self, logfile, secretary):
        # Call the parent __init__()
        super(Server, self).__init__(logfile, secretary)

        # The separator used by the server between IMAP folders and subfolders
        # Should be either . or /
        self.separator: str = ""

        self.mailbox: str = ""
        """The currently-opened or last-opened mailbox (aka (sub)folder)."""

        self.folders: list[str]
        """
        The list of all IMAP mailboxes (folders and subfolders) found on the current server. This attribute is auto-set when initializing a connection to a server. It gets refreshed when new folders are added programmatically at runtime.
        """

        self.inbox: str
        """
        The case-sensitive name of the system top-level and default mailbox. Gmail and Dovecot comply with the standard and call it `INBOX`, but Outlook/Office365 gets creative and call it `Inbox`. This attribute is properly set for the current server and should be used for portability instead of hard-coding `"INBOX"` in filters.
        """

        self.junk: str
        """The case-sensitive name of the server spam mailbox, typically called `Junk` or `Spam`."""

        self.trash: str
        """The case-sensitive name of the server trashbin mailbox."""

        self.sent: str
        """The case-sensitive name of the server mailbox where copies of sent emails are kept. Note that some client store sent emails in the same folder as the email they reply to."""

        self.archive: str
        """The case-sensitive name of the server mailbox where old emails may be automatically archived. Not all servers use it."""

        self.drafts: str
        """The case-sensitive name of the server mailbox where emails written but not yet sent may be saved."""

        self.flagged: str
        """The case-sensitive name of the server mailbox where emails marked as important (having the standard flag `\\Flagged`) may be moved or duplicated. Not all servers use it."""

        self.n_messages:int
        """Default number of emails to retrieve (starting from the most recent). Set from the `entries` config parameter."""

        self.server: str
        """URL or IP of the mailserver. Set from the `server` config parameter."""

        self.user: str
        """Username of the mail account on the mailserver."""

        self.password: str
        """Password of the mail account on the mailserver."""

        self.port: int = 993
        """Connection port on the mailserver. Defaults to 993 (IMAP SSL)."""
