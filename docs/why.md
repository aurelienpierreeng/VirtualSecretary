# Why Virtual Secretary ?

## Context & Intro

Emails are underrated.

Chats, forums and social medias all tried to replace them with something cooler : Facebook, Google chat/hangout, Slack, Matrix, Telegram, Discord, Discourse, Github… Your digital communications are now scattered over many platforms that promised you to make emails outdated. From there, you have 2 options :

1. install all the desktop & mobile apps of all the platforms you use on your mobile/desktop device, then get flooded with notifications at the worst possible time and have your batteries drained to refresh everything all the time, while still being unable to address the notifications content because typing on touch screens is a pain, and so is reading text on 5'' screens where the visual keyboard takes a quarter of the space.
2. centralize all notifications to your email box, then enjoy the paradox of all these emails created by services meant to replace them and the overwhelming of having notifications you can discard hiding important stuff requiring action.

So, trying to replace emails just failed.

Anyway, there are lots of hardly-replaceable uses for emails, like sending account credentials, password resets, unique passwords for double auth, security warnings, communicating with businesses, institutions, calendar events invites, etc. Not to mention, all those direct emails from individuals who still use them.

People are convinced emails are bad because they hinder their productivity, but it's just that they don't use them properly.

## What are emails ?

Emails are pieces of text, possibly with media attachments, sent between two users, with a subject and a date.

Realize that description applies to pretty much all electronic communications, from SMS to forum posts, blog comments, social media posts/comments/direct messages, etc. ?

The huge difference is emails are decentralized while being centralized. Decentralized because you can send messages to users who don't have an account on the same server as you (seems like the web 3.0 idiots just dicovered that now, but felt the urge to coat it with bloaty CPU-burning blockchain). Centralized because you can gather messages from many servers to a single place, avoiding to open many websites and applications to check all your channels.


## Why emails suck ?

Emails suck for 3 reasons:

1. Desktop and mobile email clients suck, as well as mobile apps,
2. 85 to 90% of emails are spam or unwanted bulk/newsletters emailing, but spam filters suck,
3. You can't decide who is allowed to send you emails.

Note that none of these reasons actually affect emails as a communication medium: what sucks is how poorly they are managed by office applications and integrated into our modern ecosystem of office tools. Imagine if someone put as much effort into a mobile email app as Whatsapp (yes, that would give Blackberry Hub)…

Basically, emails are the communication service of the digital office aka the glue between people, servers and services. What sucks in email clients is they are self-centered and don't really connect to the rest of your services, like your customers SQL database, your agenda, your contact book, etc.

The key to good email use is to filter and sort emails depending on their importance and whether they require an action from you. To achieve that, you need more than looking *inside* emails, you need to connect email content with other servers and sources to apply filters and workflows.

### Email filters

Every email client app has a way to define filters, which will allow you to write conditional actions that attribute tags, move emails to a folder, delete or reject them. It's basic but great.

Problem is, filtering can't happen at the client-level anymore, because you typically have at least one desktop and one smartphone fetching your emails through IMAP at the same time, which means at least 2 clients. Whatever client gets the emails first will decide their fate, unless you duplicate all filters across all your clients. Nobody does that, especially since mobile email apps typically don't support filters.

The only way to solve that is to have server-level filters, that run before clients get a chance to fetch emails. But there are lots of situations where you don't have control over your email server to do so.

So you have to either resort to giving up your privacy to Gmail or to use some sort of bloated and mildly-unsecure webmail or groupware, operating on a server and running filters online. But the filtering based solely on email content is still archaic and incomplete.

In real life, you need to cross-check email content with your contact book, your agenda, your project/tasks, possibly with some customer database, etc. You may also want to use regular expressions for advanced text parsing or even language processing AIs to identify the nature of the email. None of this is currently possible, in email clients as in webmails/groupwares.

### Spam filters

All the emails clients I have tested over the years (Thunderbird, Horde/Nextcloud email, EGroupware) have no simple way to permanently whitelist some sender's IP/email to prevent their spam filter to output false-positives. None of them have a way to permanently blacklist them either to prevent false-negatives. None of them can be explicitly trained by users.

SpamAssassin and such look for IP and URLs reputation, and other technical matters (SPF, DKIM) but they don't process the content (keywords) well and don't take user input (other than tweaking directly the config files, perhaps). So they let through technically-valid newsletters from mailing-lists you never subscribed to.

Then you have no easy way to unsubscribe to all that junk.

### Connecting to other services

It took Thunderbird until 2018 to finally ship Lightning, the calendar extension (supporting CalDAV and Google Agenda), within its core app. ~~As of 2022, it still doesn't provide a way to connect to CardDAV (contact books) servers~~ CardDAV support was finally added in Thunderbird 102… released in June 2022, but you need to know where to look for it because nothing says "add an address book from a server". Its internal contact book still doesn't support CardDAV features like assigning categories/groups to contacts in order to send bulk emails, for example.

Because emails are standardized, as well as CardDAV, CalDAV and WebDAV, and because email/contacts/agenda/file-sharing defines the base ground of all office work, it must connect and integrate flawlessly within each other. This is still not a reality in 2022 and is probably the reason why people endure emails.

Thunderbird is of course not the only client available around, but KMail and Evolution are even worse. The only alternative is to resort to Horde, Zimbra, SoGO and such, which are all webmails to deploy on servers, but they are tedious to install, complicated to secure, and clearly overkill for individuals and small businesses.

## Imagining a better present

We live crazy lives with too many inputs, and we get more emails than we can humanely process. In this mess, the only way to keep your sanity is to have your inputs served along with an instruction telling you what you should do with them :

* Do they require an action from you :
  * yes and urgently ?
  * yes but whenever you get the time ?
  * optionaly, if you feel like it ?
  * no, information for your archives only ?
  * yes: unsubscribe from that bloody mailing-list ?
* Who is the correspondent :
  * a friend or family ?
  * a paying customer ?
  * a prospect/future customer ?
  * a colleague or supervisor ?
  * a troll or a bully ?
  * a bot or an automated mailing system ?

In the past, people used to have secretaries to achieve this task and shield them from unwanted sollicitation, allowing them to focus and do their work efficiently while still allowing urgent matters to go through. They also sorted the mail by category and priority, tracked time, booked appointments and planned travels. The 21th century workers have to do all that by themselves, but as of 2022, the consensus is that multitasking doesn't work and only creates anxiety while destroying productivity.

So we need software to sort the mail, remove the junk and give you hint about the urgency of the mail.

## Why is Virtual Secretary better ?

1. **It connects directly to your servers.** Those are designed for concurrent access with mechanisms to prevent data corruption.
2. **It can run on server or on desktop.** It's just a script to call, you can make it a cron job.
3. **It is independent from any client.** You can keep using the clients you want, the filters are run standalone in background and don't rely on a particular client.
4. **Your configuration is a folder with meaningful text files in it.** We already have a GUI for that : your file browser.
5. **You can do a lot with few lines of code**. Connections, decoding and parsing are done backstage, just write content filters.
6. **It's still Python inside**. Train AIs, call Rest APIs, parse HTML, match regex, plot graphs, do stats, etc. with just a few imports and the 2nd most used language in the world.
7. **There is no GUI**. Yes, it's actually better for an application designed for flexibility: replace a bloated window trying to account for every marginal sub-use by just a couple of lines of code that you can <kbd>Ctrl+F</kbd> and RTFM.

## Performance

CPU use and program runtimes are important if you are going to process large amounts of emails or need to use a server sparingly. Tested on Intel Xeon E3-1505M @ 3 GHz × 8 on battery with balanced performance P-State:

* Getting the last 50 emails takes between 32 and 72 s *,
* Parsing 50 emails takes in average 1.25 s,
* Filtering 50 emails takes between 0.1 and 1.1 s depending what you are doing.

On the same laptop, plugged in AC, with balanced performance P-State:

* Getting the last 500 emails takes between 52 and 101 s,
* Parsing 500 emails takes between 6 and 8 s,
* Filtering 500 emails takes between 0.2 and 3.5 s.

Runtimes are given in the standard output in terminal.

Please note that:

1. The time needed to fetch emails depends first and foremost on your network and server,
2. The time needed to parse emails will increase if there are attachments,
3. The time spent filtering takes into account only the filter check (aka deciding whether or not each email needs to be acted-on), not the actual action. Actions that need to reach a server through a network (move email to IMAP folder, auto-reply through SMTP, ping some server through HTTP) depend first and foremost on your network and server.
4. Some data, like the email body, URLs and IPs found in email body, etc. are lazily parsed, which means that patterns are searched in body only if and when the properties are accessed. This will go in the filtering time. Only IMAP headers are parsed all the time.

## Backstory

2022 was by far my worse year, regarding emails and quantity of work. I lost many important notifications from Github regarding [darktable](https://github.com/darktable-org/darktable/) and got flooded under shit.

Between January 1st 2022 and October 1st 2022, I got 2164 notification emails linked to the darktable project, over 1200 newsletter emails (non-spam), 3190 spam emails and only 750 relevant emails trapped in-between. That's an average of more than 37.5 emails per day, of which 2.7 in average required my time and attention.

Problem is, the kind of work I do requires concentration and peace to manipulate abstract concepts and solve problems. So I don't open my mailboxes everyday. Good luck then finding those 2.7 relevant emails in the middle of the junk. That was a nightmare. Never again.

I have used Thunderbird extensively since 2007. That was long before I actually needed a mailbox. As my daily flow of emails increased, I started setting filters and such. I lost my whole Thunderbird config twice, including filters and RSS feeds, due to updates triggering file corruptions. The second time, I didn't bother redoing it from scratch… Why invest in something unreliable ?

Thunderbird has lots of features and yet manages to lack the ones I really need. I use the search engine, the "write" and "forward" buttons, and that's about it. No GPG thing since nobody uses that. Though my calendar is connected to Thunderbird/Lightning, my appointments are booked mostly by software or email invites, and I find myself checking my day planner only on my phone. So Thunderbird is a big piece of bloat that still fails to provide good filtering features and agenda/contacts bindings.
