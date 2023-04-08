The emails handling is split in 2 parts:

- the server part, dealing with IMAP connection, fetching/adding mailboxes folders etc.
- the object part, dealing with the email content per-se.s

## Preamble

There are a couple of oddities with IMAP servers that you should be aware of.

### Standards are for idiots who believe in them

Gmail and Outlook seem to take an ill-placed pleasure in ignoring IMAP standards (the RFCs), that are otherwise well supported by open-source server software like Dovecot. As such, there are several things we can't do "the simple way" because we have to account for those discrepancies. Methods are provided in the `imap_server` class to get data (like the names of the default mailboxes) properly, no matter the server implementation. You are advised to always use the provided methods to get mailboxes data for your filters, because they take care of the discrepancies internally and allow you to write portable filters.


### Mail account vs. mailbox

A mail account has usually (always ?) a root, main, default, top-level mailbox named `INBOX` or `Inbox`, depending on servers (it's case-sensitive). That's where incoming emails end up. Then, this one has subfolders, like `Sent`, `Junk`, etc., also named mailboxes by IMAP specification. That can be confusing, so I always refer to them here as "folders" and "subfolders".

IMAP servers only let you grab emails from one mailbox at a time, in a non-recursive fashion. It means that we will need to iterate over the list of known folders and subfolders to fetch all emails from a mail account. This list can be found in [protocols.imap_server.Server.folders][].

### Emails have no truly unique ID

The IMAP UID of an email is only the positional order of reception of the email in the current mailbox. When moving emails to another mailbox, their UID will actually change. But moving emails to another mailbox and back to their original mailbox will not give them back their original UID either, as it is an index that can only be incremented.

The [RFC 822](https://www.rfc-editor.org/rfc/rfc822) defines the `Message-ID` header, that is indeed an unique identifier set when sending an email, like `abcdef@mailserver.com`, where `abcdef` is a random hash. The problem is this ID is set at the discretion of the email sender, and spam/spoofed emails don't have one.

To circumvent this issue, the [protocols.imap_object.EMail.create_hash][] method creates a truly unique and persistent hash, using the data available in the email, like its date, sender and `Message-ID` header, in order to identify emails in logs through their moves between mailboxes.

Unfortunately, IMAP actions still have to use the IMAP UID.

## API

::: protocols.imap_server.Server

::: protocols.imap_object.EMail
