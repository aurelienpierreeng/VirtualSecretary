# Risk analysis

Emails, contacts and agendas usually contain personal and even sensitive information. Any application dealing with them should mention how these data are collected, stored and used such that users may assess if the security level of the application is satisfying for their use.

*Virtual Secretary* connects to IMAP, SMTP, CardDAV servers through SSL only. By design, less-secure protocols are not implemented. This means that data are encrypted during transport through the network, but are stored plain (non-encrypted) on servers.

Emails encrypted with PGP are not supported in read and write, meaning only headers will be read and no encrypted email will be written.

The data fetched from IMAP, SMTP, CardDAV servers are stored in the computer RAM for the length of the session (during the `main.py` script execution). No local copy is made on disks and stored for later use, which means all data will be re-downloaded at each run. While this slows-down execution, it avoids storing unsecure data.

The connections to IMAP, SMTP and CardDAV servers use simple passwords. This is unfortunately not very secure, though the use of SSL mitigates some risks, but many servers don't support certificates. The passwords are stored in plain text in `settings.ini` in the `config` directory. It is advised to use "application passwords" that are particular to each application, such that the access of compromised applications can be quickly revoked.

Aurélien Pierre declines any responsibility regarding loss of data and security breaches contained in or made possible by Virtual Secretary. The scripts are provided for what they are worth and without any guaranty.
