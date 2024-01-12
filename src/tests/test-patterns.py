import regex as re
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core.patterns import *
from core.utils import timeit

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

assert matches == reference

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
"""

matches = find_pattern(URL_PATTERN, samples)

[print(m) for m in matches]
