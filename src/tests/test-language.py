"""
Test the sentences and word tokenization.

"""

import os
import sys

import tests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import nlp
from core import utils

tests = [
  ( "french", "Nous pensons que me too est overrated parce que son marketing est has-been" ),
  ( "french", "Bonjour, ça va bien ?" ),
  ( "english", "She has a je ne sais quoi so charming"),
  ( "japanese", "8金その返扱マオ断中ヘ社観かじ禁提レもぴ試写ム目面鳥決ふ気患れた意宮47了軽訴防クリ。要るいむな情芸気たきべち落季めやみこ断39町者水ぴぞぎご奈手ヲオ松負ツカ恭用どゃいび同備ヌア大査ませ分94夢巻為しゆぎ。文カ被式ホハ治8村量ゆいク婚税ムケセカ化駆ょレな題環モヒテハ存呈十スセ花安ムイアツ循社接ラか定周ほもろば三副マテ読親ぴちドっ重事ル飾菌カチエヱ展盤旅伝追の yes we can"),
]

for test in tests:
  print(test[0], nlp.guess_language(test[1]))
