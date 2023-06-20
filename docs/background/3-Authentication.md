# Email authentication

The Virtual Secretary supports email authentication through [SPF](https://en.wikipedia.org/wiki/Sender_Policy_Framework), [DKIM](https://en.wikipedia.org/wiki/DomainKeys_Identified_Mail) and [ARC](https://en.wikipedia.org/wiki/Authenticated_Received_Chain).

## Why ?

Sending an email by impersonating you is incredibly easy. To do it in Python, I just need to run this snippet :

```python
import turbomail

message = turbomail.Message("you@gmail.com", "you@gmail.com", "I hacked your account")
message.plain = "I have taken control of your mailbox and found interesting things, you dirty little minx. Pay me 0.5 bitcoin now or I will send everything I found to your family."

turbomail.enqueue(message)
```

… And sent. That's it, no need for login/password. The `from` email address above is set the same as the `to` address, but it is purely declarative: the email above will be sent without needing the credentials of your actual Gmail account. This technique is used a lot by scammers who try to make you believe they hacked into your mailbox and try to ransom you.

Because it is that easy to pretend to be anybody, and because spammers and scammers do it a lot, it is necessary to have ways of authenticating first the sender of an email (aka the `from` field is authentic) but also the content of the email (aka the content didn't got tempered during transport).

### SPF

SPF allows to declare, on your mailserver, what IP addresses are allowed to actually send emails on "your" behalf, "you" being the `Return-Path` header of the email you send. Say you are `you@gmail.com`, the Virtual Secretary queries the `gmail.com` domain for the `MX` entries and checks if the email was sent by a server whose IP matches one of those and any additional one defined in the [SPF DNS entry](https://www.cloudflare.com/learning/dns/dns-records/dns-spf-record/).

SPF is fairly easy to set, as it requires a simple text entry, and has been around for quite some time. Discarding emails that fail SPF checks reliably removes 90 % of spams.

### DKIM

SPF allows to validate the origin server of the email, but does not validate neither the sender (the actual person) nor the content (integrity). DKIM solves that, with a public and private key. The public key is hosted in a DNS entry and can be retrieved from the server. The private key is typically managed by your SMTP server, the one you should login to through SSL. When an email is sent, the SMTP server will sign the headers of the email with the private key, which will produce a sort of hash. The recipient, fetching the public key on the server, can check that the hash is consistent with current headers and public key.

DKIM is a bit more difficult to setup, as it needs to be handled at 2 places (DNS server entry and SMTP mailserver), but is supported by most email services. It is a safer than SPF alone.

There are caveats though. DKIM keys are valid only for some period of time. It may happen that old emails will fail the DKIM test because the DKIM key has changed. It also happens a lot that the IT guys are sleeping on their desk and forget to renew the key after its expiration (it's currently the case with Proton mail).

For example, I'm writing this on March 28th. All emails from Github sent after the 23rd pass the DKIM check, but all emails sent prior to the 22nd fail with "no DKIM found on the DNS", which means that Github updated its keys on the 23rd but without leaving the old ones as alternatives for some more time.

### DMARC

DMARC is a policy that you can enforce on your domain to tell mailservers what they should do with emails allegedly coming from you but failing SPF and DKIM checks. When you are confident that your SPF and DKIM settings are working, you can setup DMARC to tell all mailservers around to refuse anything that appears to come from you which doesn't pass the checks, and that may help to preserve your online reputation if spammers try to impersonate you (avoiding you grey- and black-listing of your IP/domain).

DMARC is used by the mailserver to decide whether emails are accepted or will bounce (and be sent to the `Return-Path` address) when they fail the checks. When you use the Virtual Secretary, you are only parsing emails already accepted by the server through IMAP, so DMARC doesn't concern us. Also, I'm not sure how compliant most mailservers are, so we may not rely on them. The Virtual Secretary allows you to define your own actions based on the succes of the SPF and DKIM checks.

### ARC 

DKIM still has one problem: some server may transfer emails by legitimately changing their headers, which will make the DKIM check fail. This is legitimate for all cases where emails are forwarded, for example by marketing mass-mailing or mailing-lists. ARC allows to validate the forwarding step, and ARC validation can be used as a second-chance when DKIM validation has failed.

## Implementation in the Virtual Secretary

The Virtual Secretary implements a `email.authenticity_score()` method that computes a cumulative score of all ARC, DKIM and SPF checks and internally handles all the low-level parsing.

It is worth mentionning that many spammers/scammers forge fake DKIM, SPF and ARC headers, so the emails look ok from aside (all the security bits are there), but when you actually run the checks of the cryptographic keys and hashes, you find out that nothing is valid. Forged signatures get penalized heavily, with an authenticity score of `-2`. Any email getting a score of `-6` has all DKIM, SPF, and ARC headers forged.

Explicit success is awarded a score of `+2`. That is, when we have an header and it passes the check.

When no header is found, or, for SPF, when the SPF rule doesn't explicitly disallow anything, the score is `0`.

For real emails, I got the following scores:

* Youtube, Gmail : `+6` (valid ARC, explicitly valid SPF, valid DKIM),
* PayPal, Stripe, most banks since 2019, most email providers : `+4` (no ARC, explicitly valid SPF, valid DKIM),
* Github, Proton Mail : `+3` (no ARC, explicitely valid SPF, outdated DKIM signature),
* Semi-serious marketing emails from legit companies, banks before 2018, self-hosted emails with some IT skills : `+2` (no ARC, no DKIM, explicitly valid SPF),
* your uncle Bob who setup a mailserver in the basement, your nephew Jason who got a shared hosting without spending quality time on CPanel : `0` (no ARC, no DKIM, no explicit SPF rule),
* my grand-father who pushes all buttons without understanding what they do : `-1` (no ARC, no DKIM, bad/misformed SPF rule),
* anybody trying to play you a nasty one : `-2` to `-6` (forged headers).

Note that, in cases where emails have been forwarded, the DKIM header will fail with a score of `-2` but if the ARC check passes, the total score will be zero. Along with a valid SPF, that should still give a total greater than zero.

You can use the `email.authenticity_score()` method directly from your filters, otherwise a convenience `email.is_authentic()` method is provided and returns `True` for scores above zero.

Internally, the `email.authenticity_score()` method calls each check and adds their output score. You can also use them in your filter, they are `email.spf_pass()`, `email.dkim_pass()` and `email.arc_pass()`.

## Efficiency

Discarding emails based on their authenticity score has proven to catch roughlà 98 % of spams, phising, spoofings and such, which is much better than any keyword-based content filtering. If you also check for emails which has the `Precendence: bulk` header (mass-mailing) without having a `List-Unsubscribe` (allowing for easy unsubscribe from mailing-lists), you get most of the nasty emails out of your box. Note that Gmail doesn't seem to do it.

The drawback is authenticity needs to ping several DNS for each email (to get SPF rules, MX entries, DKIM key, etc.), which needs an internet connection, needs some time, and will take forever if some DNS stutter.

Running authenticity checks should be done __before__ checking if the sender belongs to your contacts book to ensure the email actually comes from your contact.
