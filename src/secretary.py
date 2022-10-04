from asyncio import protocols
import mailserver
import utils

import configparser
import os

class Secretary(object):
  def load_connectors(self):
    for key in self.protocols:
      if key in self.config_file and key in self.protocols:
        self.protocols[key].init_connection(self.config_file[key]["server"],
                                            self.config_file[key]["user"],
                                            self.config_file[key]["password"],
                                            int(self.config_file[key]["entries"]))
  def close_connectors(self):
    for key in self.protocols:
      if key in self.config_file:
        self.protocols[key].close_connection()

  def filters(self, filters:dict):
    # Launch the method that processes the filter loop for this server
    for key in sorted(filters.keys()):
      filter = filters[key]["filter"]
      filter_path = filters[key]["path"]
      protocol = filters[key]["protocol"]

      # Check that we have a server and an active connection for this protocol
      server_instance = self.protocols[protocol]
      if not (server_instance and server_instance.connection_inited):
        print("We have no active connector for the protocol %s, check that you defined your credentials in `settings.ini` for it." % protocol)
        return

      with open(filter_path) as f:
        print("\nExecuting filter %s :" % filter)
        self.logfile.write("%s : Executing filter %s\n" % (utils.now(), filter))
        code = compile(f.read(), filter_path, 'exec')
        exec(code, {"mailserver": server_instance, "filtername": filter_path})

  def __init__(self, subfolder_path:str):
    # Unpack the servers credentials
    self.config_file = configparser.ConfigParser()
    self.config_file.read(os.path.join(subfolder_path, "settings.ini"))

    # Start the logile
    self.logfile = open(os.path.join(subfolder_path, "sync.log"), 'a')

    ## Open all servers for which we have credentials in config file
    # Declare the list of all supported protocols and instanciate their server classes here.
    # Keys should match the sections of config files and the protocol prefix in filter filenames.
    self.protocols = {}
    self.protocols["imap"] = mailserver.MailServer(self.logfile)

    # Load all implemented connectors
    self.load_connectors()
