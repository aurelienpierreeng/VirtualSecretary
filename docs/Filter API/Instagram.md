Instagram has grown into a major pain in the ass since it went from a casual Instamatix-like microblogging app to a de-facto business-oriented influencer/influencee advertising platform that still wants to rely 100% on the mobile app paradigm. While this makes it practically a must-have for businesses and brands online presence, managing customers through IG direct messages diverts your work into one more channel, which is, by design, not integrated into the rest of your tools. Mix that with Twitter and YouTube and you have all you need to spend the day answering messages without getting anything done.

While Instagram allows you to get emails notifications for new comments, you can't get direct messages (DM) as well (which makes zero sense from an usability perspective). I will present here a *Virtual Secretary* workflow to duplicate comments and DM from Instagram into your IMAP mailbox.

# Requirements

This uses the [Facebook Graph API for Instagram](https://developers.facebook.com/docs/instagram-api), which imposes that:

1. You have one Instagram [Business](https://help.instagram.com/502981923235522) or [Creator](https://help.instagram.com/2358103564437429) account (aka not the regular customer account),
2. You have a Facebook page for your business,
3. You have a Facebook account that has admin rights on that page,
4. Your Facebook business page is [linked](https://help.instagram.com/570895513091465) to your Instagram Businesse/Creator account,
5. You need to [create an app](https://developers.facebook.com/apps/create/) for your business in order to allow the members of your organization to use it (it will not be published for general use):
    1. Under "type", choose ["Business"](https://developers.facebook.com/docs/development/create-an-app/app-dashboard/app-types) (as of 2022, API v15). In any case, you need the account type allowing access to Graph API for Instagram and Messenger,
    2. Choose the Business account linked to your Facebook page,
    3. Write down the app ID, which will be used in the next step inside the config assistant.
6. You need to enable the [message control by connected apps](https://developers.facebook.com/docs/messenger-platform/instagram/get-started#connected-tools-toggle) in the Instagramm app, under Settings -> Privacy -> Messages.

Note that:

1. Accessing your IG DMs is actually done through your FB account, so you need both accounts alive,
2. Linking your FB page to your IG account can only be achieved through the IG mobile app, not on the website,
3. Linking your IG account to your FB page **from** your FB page doesn't work, it needs to be done on the IG mobile app.
4. "Creating an app" is the FB/IG way of saying "getting a Rest API key", because they assume that you want the first if you are requesting the other. All we need is to generate an oAuth token by which you can grab IG user data from IG from a script. Facebook will nag you about getting your app reviewed assuming you want to publish it for general use (that is, grabbing **other user's** data), but as long as only the "app" owner uses it to access his own account, there is no need.

# Set-up

1. Install the Python dependencies for the config assistant): 
```python
pip install PySide6 oauthlib requests_oauthlib
```
2. Call `python src/facebook-login.py`. This will prompt an user-friendly config assistant allowing you to connect to your FB account to grant access to your app and setup permissions. You need to grant all requested permissions on your pages for your app. This practically means getting an authentification token, along with the ID of your FB page and IG account. 
3. If everything goes well, a new config file called `instagram.ini` will be created in the same folder as your `src/facebook-login.py`, with everything we need, looking like :

```ini
[instagram]
access_token = XXXXXXX
data_access_expiration_time = 1675362365
expires_in = 0
state = EWZw0pzq8bLmC6PqQ4B20L66MHH7vg
expires_at = 1667586366.0085597
client_id = XXXXXX
business_id = XXXXXXX
instagram_business_account = XXXXXX
```

4. Copy and paste the content of the `instagram.ini` file above in the `settings.ini` file of the relevant email directory. If your *Virtual Secretary* instance runs on a remote server, you can generate the token locally on your desktop and copy-paste the IG credentials on your server. 
5. This token is valid only for 4 months (typically), so you will need to renew it in the future following the same procedure.

Notes:

* The config assistant aims at providing a basic yet user-friendly way of retrieving the IDs of accounts. It does not check all possible corner cases, does not protect users from themselves and works only if all the requirements above have been met. If it doesn't work, you have messed up somewhere and need to check that your FB account has the proper management permissions over the FB business page, the FB business page is properly linked to an IG business account, etc.
* The config assistant runs locally on your device, fetching authentification token for your app with your account. Everything happens between you and Facebook, no third-party service provider or server is involved.
* The access token fetched by the assistant is like a password, you need to store it securely, not post it in forums, questions or post screenshots containing it.
* If there is already an `instagram.ini` file along the script `/src/facebook-login.php` and the authentification token is still valid, the config assistant uses it by default. If you need to configure another account, you will need to rename, move or delete this file.

# Use

TODO.