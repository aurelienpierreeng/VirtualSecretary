import regex as re
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core.patterns import *
from core.utils import timeit

tokens = ["2045", "e62fabc2", "2.5", "4.999.23"]

for token in tokens:
    hash_result = HASH_PATTERN_FAST.match(token)
    number_result = NUMBER_PATTERN_FAST.match(token)
    if hash_result:
        print("hash match", hash_result.group(0))
    if number_result:
        print("number match", number_result.group(0))


@timeit(runs=100000)
def find_pattern(pattern: re.Pattern, string: str):
    return re.findall(pattern, string)


samples = """
Should be caught:
- 12h15
- 12:15
- 12:15:00
- 12am
- 12 am
- 12 h
- 12:15:00Z
- 12:15:00+01
- 12:15:00 UTC+1
- 6:45 am
- 6:45am
- 6 am
- 6am
- 24 h
- 12 h 12
- 12:45 UTC+2
- 12:45Z
- 14:45+01
- 14:45:00-02
- 15H
- Added:\n2023-03-21T11:27:45+0000

Should be ignored:
- 1
- 23
- 45 $
- 12 : 01
- 2022-12-01
- 1 €
"""

matches = find_pattern(TIME_PATTERN, samples)

reference = [('12', 'h', '15', '', '', ''), ('12', ':', '15', '', '', ''), ('12', ':', '15', '00', '', ''), ('12', 'am', '', '', '', ''), ('12', 'am', '', '', '', ''), ('12', 'h', '', '', '', ''), ('12', ':', '15', '00', 'Z', ''), ('12', ':', '15', '00', '', '+01'), ('12', ':', '15', '00', 'UTC', '+1'), ('6', ':', '45', '', 'am', ''), ('6', ':', '45', '', 'am', ''), ('6', 'am', '', '', '', ''), ('6', 'am', '', '', '', ''), ('24', 'h', '', '', '', ''), ('12', 'h', '12', '', '', ''), ('12', ':', '45', '', 'UTC', '+2'), ('12', ':', '45', '', 'Z', ''), ('14', ':', '45', '', '', '+01'), ('14', ':', '45', '00', '', '-02'), ('15', 'H', '', '', '', ''), ('12', ':', '01', '', '', '')]

#assert matches == reference

[print(m) for m in matches]

samples = """
https://moi.com
https://moi.com/
https://moi.com/page
https://moi.com/page/
http://domain.ext/page/subpage
http://domain.ext/page/subpage/
http://domain.ext/page/subpage/#stuff
http://domain.ext/page/subpage/?q=x&r=0:1+i
http://domain.ext/page/subpage/?q=x&r=0:1+i#stuff
http://domain.ext/page/subpage/#stuff?q=x&r=0:1+i
Stuffhttps://moi.com blah
Stuff https://moi.com blah
Stuff https://moi.com/ blah
Stuff https://moi.com/page blah
Stuff https://moi.com/page/ blah
Stuff http://domain.ext/page/subpage blah
Stuff http://domain.ext/page/subpage/ blah
Stuff http://domain.ext/page/subpage/#stuff blah
Stuff http://domain.ext/page/subpage/?q=x&r=0:1+i blah
Stuff http://domain.ext/page/subpage/?q=x&r=0:1+i#stuff blah
Stuff http://domain.ext/page/subpage/#stuff?q=x&r=0:1+i blah
blah
ftp://ftp.me.com/page
Stuff: ftp://ftp.me.com/page blah
://stuff.me
/home/stuff
//meh.photo
blah blah eh http://domain.ext/page/subpage/#stuff?q=x&r=0:1+i
Blah blahhh <a href="https://at.dot.com/page">Blah</a>.
Blah blahhh <a href="https://at.dot.com/page/">Blah</a>.
Blah [link](https://at.dot.com/page)
Blah [link](https://at.dot.com/page/)
Meh [https://at.dot.com/page]
https://web.archive.org/web/20181103230658im_/http://atelier-malopelli.com/wp-content/uploads/2016/03/Universelle-Sol-Re%CC%81-2-.jpg
http://localhost:2080
<a href="https://web.archive.org/web/20181103230658im_/http://atelier-malopelli.com/wp-content/uploads/2016/03/Universelle-Sol-Re%CC%81-2-.jpg">
<http://domain.ext/page/subpage/?q=x&r=0:1+i#stuff>
<title>http://domain.ext/page/subpage/?q=x&r=0:1+i#stuff</title>
https://cuchara.photography/2019/05/better-negative-scans-using-flat-field-correction-in-lightroom/)
"""

matches = find_pattern(URL_PATTERN, samples)

[print(m) for m in matches]

samples = """
2001:0db8:0000:85a3:0000:0000:ac1f:8001
2001:db8:0:85a3:0:0:ac1f:8001
2001:db8:0:85a3::ac1f:8001
2001:db8:1:1a0::/59
2001:db8:1:1a0:0:0:0:0
2001:db8:1:1bf:ffff:ffff:ffff:ffff
2001:41D0:1:2E4e::/64
2001:41D0:1:2E4e::1

192.168.1.1
172.0.0.0
79.241.182.32
92.123.25.32

12h15
12:15
12:15:00
12am
12 am
12 h
12:15:00Z
12:15:00+01
12:15:00 UTC+1
"""

matches = find_pattern(IP_PATTERN, samples)

[print(m) for m in matches]


samples = """
13€
13 €
13.54 €
€13.54
€ 13.54
50.2k€
50.2 k€
1 000 €
1 000€
€1 000
1,000.54 €
52,52 €
+50 €
-200€
"""

matches = find_pattern(PRICE_PATTERN, samples)

[print(m) for m in matches]


samples = """
50mm f/1.8 G
50mm f/22
"""

matches = find_pattern(DIAPHRAGM, samples) + find_pattern(DISTANCE, samples)

[print(m) for m in matches]


samples = """
20%
25 %
26.1 %
27.5%
21
"""

matches = find_pattern(PERCENT, samples)

[print(m) for m in matches]

samples = """
Shift+Click
Ctrl+Click
Shift+Maj+E
tab + t
ctrl + tab
cmd + shift + f1
"""

matches = find_pattern(SHORTCUT_PATTERN, samples)

[print(m) for m in matches]

samples = r"""
C:\\windows\stuff\doc.txt
https://company.com
~/images/stuff/i.jpg
/home/user/stuff
./subdir/here
f/8
f/1.8
"""

matches = find_pattern(PATH_PATTERN, samples)

[print(m) for m in matches]


samples = """
stuf
stuff
stufff
"""

matches = find_pattern(INTERNAL_NEWLINE, samples)

[print(m) for m in matches]
