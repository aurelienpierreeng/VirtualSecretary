import connectors
import utils
import requests
import vobject

from requests.auth import HTTPBasicAuth
from xml.dom import minidom


class Card(connectors.Content):

  def __getitem__(self, key):
    # Getting key from the class is dispatched directly to email.EmailMessage properties
    output = []
    if key in self.vcard.contents:
      for elem in self.vcard.contents[key]:
        output.append(elem.value)

    if len(output) > 0 and isinstance(output[0], list):
      # Categories are stored as a list of lists for no reason.
      # We want a flat 1D list, so we need to fix that.
      output = [item for sublist in output for item in sublist]

    return output


  def __init__(self, dom, server):
    self.server = server

    # Parse the XML response of the server
    self.etag = dom.getElementsByTagName('d:getetag')[0].firstChild.data
    self.href = dom.getElementsByTagName('d:getetag')[0].firstChild.data
    self.vcard = vobject.readOne(dom.getElementsByTagName('card:address-data')[0].firstChild.data)

    #print(self["fn"], self["categories"], self["email"], self["tel"])


class Server(connectors.Server[Card]):

    def get_objects(self):
        REPORTFIND = """<card:addressbook-query xmlns:d="DAV:" xmlns:card="urn:ietf:params:xml:ns:carddav">
                            <d:prop>
                                <d:getetag />
                                <card:address-data />
                            </d:prop>
                        </card:addressbook-query>"""

        result = self.session.request(method="REPORT", url=self.server, headers={"depth": "1" }, data=REPORTFIND)

        # Parse XML responses
        dom = minidom.parseString(result.text.encode('ascii', 'xmlcharrefreplace'))
        responses = dom.getElementsByTagName('d:response')
        [self.objects.append(Card(response, self)) for response in responses]


    def run_filters(self, filter, action, runs=1):
        pass


    def search_by_email(self, query:str):
        # Find all the contact vcards whose email matches the query
        result = []
        for elem in self.objects:
          for email in elem["email"]:
            if email == query:
              result.append(elem)

        return None if len(result) == 0 else result

    def get_emails_list(self):
        # Get a flat list of all emails in contacts
        # This is to be used mostly by spam filters to check wether "email" is in the list or not
        # but efficiently.
        self.emails = []
        for elem in self.objects:
          for email in elem["email"]:
            self.emails.append(email)


    def init_connection(self, params: dict):
        # High-level method to login to a server in one shot
        # See https://sabre.io/dav/building-a-carddav-client/
        password = params["password"]
        self.server = params["server"]
        self.user = params["user"]

        # Init a session with authentification on
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(self.user, password)

        PROPFIND = """<d:propfind xmlns:d="DAV:" xmlns:cs="http://calendarserver.org/ns/">
                        <d:prop>
                          <d:displayname />
                          <cs:getctag />
                        </d:prop>
                      </d:propfind>"""

        # Get the properties of the contact book
        result = self.session.request(method="PROPFIND", url=self.server, headers={"depth": "0" }, data=PROPFIND)

        if result.status_code == 207:
          # Parse the XML response
          dom = minidom.parseString(result.text.encode('ascii', 'xmlcharrefreplace'))
          response = dom.getElementsByTagName('d:response')[0]
          prop_stat = response.getElementsByTagName('d:propstat')[0]
          prop = prop_stat.getElementsByTagName('d:prop')[0]

          # Get our properties and save them
          self.href = response.getElementsByTagName('d:href')[0].firstChild.data
          self.ctag = prop.getElementsByTagName('x1:getctag')[0].firstChild.data
          self.display_name = prop.getElementsByTagName('d:displayname')[0].firstChild.data

          #print(result.text)
          #print(self.href, self.ctag, self.display_name)

          self.connection_inited = True
          self.get_objects()
          self.get_emails_list()


    def close_connection(self):
        # High-level method to logout from a server
        pass
