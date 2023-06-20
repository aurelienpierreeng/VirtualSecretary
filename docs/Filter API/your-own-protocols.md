# Implementing your own protocels

The whole logic of the Virtual Secretary is meant to allow low-overhead extensibility. Adding new filters is as easy as adding new Python files, provided they respect the proper naming pattern, and so is adding new connectors for new protocols.

## A foreword on the main logic

The structure of the program follows the [dependency inversion principle](https://en.wikipedia.org/wiki/Dependency_inversion_principle) where the interface layer is the `src/connectors.py` module which declares the abstract classes to be implemented by all protocol connectors.

The `main.py` script is the entry point of the program and will build the list of all filters detected in subfolders that match the required naming patterns. For now, only patterns matching `00-protocol-filtername.py` or `LEARN-protocol-filtername.py` (where `protocol` is the *actual* protocol you are using, like `imap`, `smtp`, `carddav`, etc.) are recognized and will be run respectively by passing `process` or `learn` mode arguments to the bash call `$ python main.py config/ mode`. Adding more options to select a different set of filters is possible in the future simply by changing the filename patterns used to build the dictionary of filters at the very beginning of the program, the core will remain unchanged.

With this list of filters to process, `main.py` calls the main manager `Secretary`, which is the class in `src/secretary.py`. This manager loads every module in `src/protocols` that has a filename matching the pattern `protocol_server.py` (where, again, `protocol` is the *actual* protocol you are using, like `imap`, `smtp`, `carddav`, etc.), start the corresponding servers, dispatch the credentials from the `settings.ini` configuration file, trigger the logins, loop through all the planned filters and then close the servers.

The `Secretary` manager knows protocol implementations only through their abstract base classes in the interface layer `src/connectors.py` and loads all the matching modules blindly. This means that the connectors must properly inherit the `Server` and `Content` base classes and implement all of their mandatory abstract methods. Connectors must also be declared in `src/protocols` within files matching the pattern `protocol_server.py`. In return, this rigid logic allows to add as many connectors as you want by just adding new modules in `/src/protocols`, and the base files take care of starting, connecting and wiring everything in about 50 LoC each.

The `Secretary` manager tracks the list of all active servers/protocols in its `Secretary.protocols` dictionary, which is passed as a global variable to all filters, allowing users to dispatch events between servers/protocols.

## Adding your own protocol

Let's say you want to add the support for FTP protocol. First, you need to add an `ftp` section in `settings.ini`, like:

```ini
[ftp]
server = ftp.server.com
user = me
password = XXXXXXXX
```

Then, you need to create `ftp_server.py` and `ftp_object` in `src/protocols`. `ftp_server.py` will need to contain at very least:
```python
from protocols import ftp_object
import connectors

class Server(connectors.Server[ftp_object.Content]):

    def get_objects(self):
        # Fetch objects on server and populate the list
        i = 0
        for element in server_queue:
            self.objects.append(ftp_object.Content(element, i, self))
            i += 1

    def run_filters(self, filter, action, runs=1):
        # Define the log file as an hidden file inheriting the filter filename
        filtername = self.calling_file()
        directory = os.path.dirname(filtername)
        basename = os.path.basename(filtername)
        filter_logfile = os.path.join(directory, "." + basename + ".log")
        # Do what you will with the logfile to track what happened here

        # Loop over objects to apply filters and actions
        for object in self.objects:
            try:
                filter_on = filter(object)
            except:
                filter_on = False

            if filter_on:
                try:
                    action(object)
                except:
                    pass

    def init_connection(self, params: dict):
        # High-level method to login to a server in one shot
        # `params` are dispatched from the `[ftp]` section of `settings.ini` file by `Secretary`
        password = params["password"]
        server = params["server"]
        user = params["user"]

        # Open connection, probably through some low-level Python lib

        # Notify that we have a server with an active connection
        self.connection_inited = True

    def close_connection(self):
        # High-level method to logout from a server
```

For more information, look into the `Server` class in `src/connectors.py`. Filters running on triggers which server has `Server.connection_inited` set to `False` are automatically discarded, so don't forget to set it but only after you got the server response to login commands. If you forgot the implement some mandatory methods, the program will throw an error before even starting a connection. You can also define read-only or write-only filters, for example the SMTP protocol (emails sending) is write-only and would have no `get_objects()` method and its `run_filters()` method would only contain `pass`.

Then, the `ftp_object.py` file will need to contain at very least:
```python
import connectors

class Content(connectors.Content):
  def __init__(self, raw_content, index:int, server) -> None:
    super().__init__(raw_content, index, server)
    # Unpack, parse, decode the raw_content
```

The `Content` class is the one that is presented to the user when writing filters, so it should contain all the relevant data pre-parsed in members as well as proxy methods to trigger usual operations from the server on this particular object, like move, delete, tag. Then, the `Server` class should wrap low-level libraries into nice, specialized, user actions.

The `Content` objects are meant to represent the basic individual data unit from the protocol you use, for example, for IMAP, it is emails, for CardDAV, it is contacts, for CalDAV, it is events, for FTP, it is files, etc. Then, in the `Server.objects` list, we aggregate them per folder, or per category, etc.

From there, you can directly start writing filters. Nothing more is needed. For nicer filter debugging, you may want to manage exceptions and print appropriate warnings and errors.

For a complete example, look at the `imap` protocol implementation.

Note that protocols must have the same lowercase name everywhere to load modules, servers and settings properly.
