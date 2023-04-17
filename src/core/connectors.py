"""
Provide the abstract classes for server and content, that need to be implemented for each protocol.

Filters and actions are applied by the `Server` by looping over each `Content` object and executing the user-defined functions.

The core of the framework knows only these abstract classes.

© 2022-2023 - Aurélien Pierre
"""

from __future__ import annotations
import typing
import inspect
from core import nlp
from core import utils

from abc import ABC, abstractmethod


class Content(ABC):
  """The `Content` class manages the decoding and parsing of individual data objects (such as posts, emails, contacts, events, etc.), as well as the data representation to be used in filters. It provides proxy methods to the `Server` class that can be used to perform server-side actions on the current `Content` object.
  """

  def __init__(self, raw_content, server: Server) -> None:
    """
    Arguments:
      raw_content (any): data dump from server that should be decoded, parsed and split into properties.
      server (Server): back-reference to the `connectors.Server` instance to which this `Content` belongs.
    """

    self.server: Server = server
    """
    Back-reference to the `Server` object to which the current `Content` belongs. In filters, users only access the `Content` object, so the `Content.server` reference is their only entry-point to the server. Use that with care and preferably wrap relevant server-side operations as `Content` methods, with proper arguments mapping.
    """

ContentType = typing.TypeVar("ContentType", bound=Content)
ServerType = typing.Generic[ContentType]

class Server(ABC, ServerType):
  """The `Server` class manages the connection to a distant host with URL, port, login and password.
  Then it loads a list of individual `objects`, like emails or contacts, as `Content` type.
  """

  nlp = nlp
  """Reference the [natural language processing module][core.nlp] for reuse in filters without imports."""

  utils = utils
  """Reference the [utility module][core.utils] for reuse in filters without imports."""

  def calling_file(self) -> str:
    """Output the name of the calling filter from the stack. Allows to find the path of the folder containing filter, to perform actions like writing logs.

    It needs to be called from filter script through object proxy method, pointing to the server method, otherwise the index 2 will be wrong.
    """
    return inspect.stack()[2].filename

  @abstractmethod
  def init_connection(self, params:dict) -> None:
    """Init a connection with the server and login.

    This should set `self.connection_inited = True` on success.
    """
    raise NotImplementedError

  @abstractmethod
  def close_connection(self) -> None:
    """Close the server connection"""
    raise NotImplementedError

  @abstractmethod
  def run_filters(self, filter: typing.Callable, action: typing.Callable, runs:int = 1, **kwargs) -> None:
    """Loop over `self.objects` to execute `filter()` and `action()` on each."""
    raise NotImplementedError

  def __init__(self, logfile, secretary: 'secretary.Secretary') -> None:
    # This is the global logfile, `sync.log`
    # Not to be confused with the local filter DB.
    self.logfile = logfile

    self.secretary = secretary
    """Back-reference to the main [secretary.Secretary][] handler"""

    self.connection_inited: bool = False
    """Set to `True` in server implementation when a connection to the server has been inited, that is the DNS/IP has been resolved, server has been contacted and responded, credentials have been sent, and valid login session is active.
    """

    self.std_out: list = []
    """Cast status/success messages under the form `["OK", "details/data"]`"""

    self.objects: list[ContentType] = []
    """The iterable list of `connectors.Content` object fetched at runtime, upon which filters will be run in sequence.
    """

    self.connection_timestamp = 0
    """Timestamp of the last successful login, useful to keep connections alive.

    Todo:
      implement it
    """
