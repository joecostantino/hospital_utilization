#!/usr/bin/env python
# coding: utf-8

# In[11]:


target_url = "https://s3.amazonaws.com/cus-615-waiting-times/file/waiting_times.txt"
import urllib3

http = urllib3.PoolManager()
response = http.request('GET', target_url)
data = response
for line in data: # files are iterable
    print(line)


# In[85]:


import urllib.request
import re

target_url = "https://s3.amazonaws.com/cus-615-waiting-times/file/waiting_times.txt"

fp = urllib.request.urlopen(target_url)
mybytes = fp.readlines()

i=0
while i <= len(mybytes)-1:
  
  find_date = "DATE:\s(\d*)"
  find_wait = '.*__numeric">(N/A|\d+)</span.*'
  find_locl = '_blank">(.*)</a>'
  next_record = "--"

  
  line_dict = []
  
  if re.findall(r'\b' + find_date, mybytes[i].decode('utf-8').strip(), re.IGNORECASE): 
      match_date = re.search(find_date, mybytes[i].decode('utf-8').strip())
      #print(i, "Full match: %s" % (match_date.group(1)))
  elif re.findall(r'\b' + find_wait, mybytes[i].decode('utf-8').strip(), re.IGNORECASE): 
      match_wait = re.search(find_wait, mybytes[i].decode('utf-8').strip())
      #print(i, match_date.group(1), "Full match: %s" % (match_wait.group(1)))
  elif re.findall(r'\b' + find_locl, mybytes[i].decode('utf-8').strip(), re.IGNORECASE): 
      match_locl = re.search(find_locl, mybytes[i].decode('utf-8').strip())
      #print(i, match_date.group(1), "Full match: %s" % (match_locl.group(1)))
  else:
      print("%s,%s,%s" % (match_date.group(1), match_wait.group(1), match_locl.group(1)))
  i = i+1

