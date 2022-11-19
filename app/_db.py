import sqlite3
import os

if os.path.exists('toxic.sqlite'):
    os.remove('toxic.sqlite')

conn = sqlite3.connect('toxic.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE toxic_db (tweet TEXT, toxic INTEGER)')

conn.commit()
conn.close()
