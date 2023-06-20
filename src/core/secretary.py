"""
High level manager of all server connectors and bank of filters:

- load available connector modules,
- pass them the content of the relevant `settings.ini` section,
- open and close the remote connections to servers by calling internal connector methods,
- process all filters of a given config subfolder in sequence by calling the internal `run_filters` method of each connector.

© 2022-2023 Aurélien Pierre
"""

import configparser
import os
import io
import typing
import importlib
import time

from core import nlp
from core import connectors
from core import utils
import protocols as prt

class Secretary(object):
  """
  Backbone class managing the collection of available connectors. It is called from `src/main.py`.
  """

  def load_connectors(self):
    """
    Iterate over all connectors for which we have both an implementation and credentials in the congig file.
    Initialize them and append them to the dictionnary [protocols][]
    """
    for key in self.protocols:
      if key in self.config_file and key in self.protocols:
        self.protocols[key].init_connection(self.config_file[key])

  def close_connectors(self):
    """
    Iterate over all initialized connectors and close the connections.
    Then close the logfile.
    """
    for key in self.protocols:
      if key in self.config_file:
        self.protocols[key].close_connection()

    self.logfile.close()

  def filters(self, filters: utils.filter_bank):
    """
    Process the loop over `Content` objects from the implementation of [connectors.Server.run_filters][] for each filter.

    Arguments:
      filters (utils.filter_bank): iterable of available filters. See [utils.filter_bank][].
    """
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


  def server_breathe(self):
    """If server mode is enabled, introduce a 0.5 s timeout to let other processes run.
    This is useful where heavy text parsing happens
    """
    if self.server_mode:
      time.sleep(0.5)


  def __init__(self, subfolder_path:str, server_mode: bool, number: int, force: bool):
    """
    Load configuration from `settings.ini` file in the current folder.
    Load all connector modules from `protocols`.

    Arguments:
      subfolder_path (str): the current folder of filters
      server_mode: enable or disable the server mode, which makes processing slower to better share limited resources on a server.
      number: override the number of items to process defined in config file. `-1` honors config files settings.
      force: ignore logs and reprocess items already processed.

    Attributes:
      protocols (dict of str: connectors.Server): available [connectors.Server][] implementations for server protocols. They are exposed to user filters in `globals()`.

      config_file (configparser.ConfigParser): object handling the `settings.ini` content for the current config folder.

      log_file (io.TextIOWrapper): object handling the main `sync.log` for the whole application, where every action on [connectors.Content][] will be logged.
    """

    self.protocols: typing.Dict[str, connectors.Server] = { }
    self.server_mode = server_mode
    self.number = number
    self.force = force

    self.config_file = configparser.ConfigParser()
    self.config_file.read(os.path.join(subfolder_path, "settings.ini"))

    # Start the logile
    self.logfile: io.TextIOWrapper = open(os.path.join(subfolder_path, "sync.log"), 'a')

    # Open all servers for which we have a connector implemented
    for file in prt.__all__:
      if file.endswith("_server"):
        protocol = file.replace("_server", "")
        protocol_module = importlib.import_module("." + file, package="protocols")
        self.protocols[protocol] = protocol_module.Server(self.logfile, self)

    # Connect to all servers for which we have credentials in settings.ini
    self.load_connectors()
