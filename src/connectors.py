"""

Provide the abstract classes for server and content, that need to be implemented for each protocol.

The `Server` class manages the connection to a distant host with URL, port, login and password.
Then it loads a list of individual `objects`, like emails or contacts, as `Content` type.

The `Content` class manages the decoding and parsing of individual objects, as well as the data
representation to be used in filters. It provides proxy methods to the `Server` class that can be
used as actions on the current `Content` object.

Filters and actions are applied by the `Server` by looping over each `Content` object and executing
the user-defined functions.

"""

from __future__ import annotations
import typing
import inspect
from abc import ABC, abstractmethod


class Content(ABC):
  server: 'Server'

  def __init__(self, raw_content, index:int, server:'Server') -> None:
    # Position of the email in the mailserver list
    self.index = index
    self.server = server


_ContentType = typing.TypeVar("Content", bound=Content)


class Server(ABC, typing.Generic[_ContentType]):
  connection_inited: bool = False
  std_out: list
  objects: list[_ContentType]

  def calling_file(self):
    # Output the name of the calling filter from the stack
    # Needs to be called from filter script through object proxy method,
    # pointing to the server method, otherwise the index 2 will be wrong.
    return inspect.stack()[2].filename

  @abstractmethod
  def init_connection(self, params:dict) -> None:
    raise NotImplementedError

  @abstractmethod
  def close_connection(self) -> None:
    raise NotImplementedError

  @abstractmethod
  def run_filters(self, filter: typing.Callable, action: typing.Callable, runs:int = 1, **kwargs) -> None:
    # Loop over the list of content
    raise NotImplementedError

  def __init__(self, logfile) -> None:
    # This is the global logfile, `sync.log`
    # Not to be confused with the local filter file.
    self.logfile = logfile

    # Init an output pipe to globally fetch IMAP commands output
    # Each internal IMAP method will post its output on it, so
    # users don't have to return the out code in the filters
    self.std_out = [ ]
