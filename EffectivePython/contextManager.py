import time

class Timer: 
    
    def __init__(self):
        self.start = 0
        self.end = 0
        
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        
import http.client as httpcaesar

with Timer() as t:
    conn = httpcaesar.HTTPConnection('google.com')
    conn.request('GET', '/')

print('Request took %.03f sec.' % t.interval)