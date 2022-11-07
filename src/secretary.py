import configparser
import os
import typing
import importlib

import connectors
import utils
import protocols as prt

class Secretary(object):
  protocols: typing.Dict[str, connectors.Server] = { }

  def load_connectors(self):
    for key in self.protocols:
      if key in self.config_file and key in self.protocols:
        self.protocols[key].init_connection(self.config_file[key])

  def close_connectors(self):
    for key in self.protocols:
      if key in self.config_file:
        self.protocols[key].close_connection()

    self.logfile.close()

  def filters(self, filters:dict):
    # Launch the method that processes the filter loop for this server
    for key in sorted(filters.keys()):
      filter = filters[key]["filter"]
      filter_path = filters[key]["path"]
      protocol = filters[key]["protocol"]

      # Check that we have a server and an active connection for this protocol
      server_instance = self.protocols[protocol] if protocol in self.protocols else None
      if not (server_instance and server_instance.connection_inited):
        print("We have no active connector for the protocol %s, check that you defined your credentials in `settings.ini` for it." % protocol)
        return

      with open(filter_path) as f:
        print("\nExecuting filter %s :" % filter)
        self.logfile.write("%s : Executing filter %s\n" % (utils.now(), filter))
        code = compile(f.read(), filter_path, 'exec')
        exec(code, self.protocols, {"filtername": filter_path})


  def __init__(self, subfolder_path:str):
    # Unpack the servers credentials
    self.config_file = configparser.ConfigParser()
    self.config_file.read(os.path.join(subfolder_path, "settings.ini"))

    # Start the logile
    self.logfile = open(os.path.join(subfolder_path, "sync.log"), 'a')

    # Open all servers for which we have a connector implemented
    for file in prt.__all__:
      if file.endswith("_server"):
        protocol = file.replace("_server", "")
        protocol_module = importlib.import_module("." + file, package="protocols")
        self.protocols[protocol] = protocol_module.Server(self.logfile, self)

    # Connect to all servers for which we have credentials in settings.ini
    self.load_connectors()
