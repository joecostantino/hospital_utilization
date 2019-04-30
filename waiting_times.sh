#!/bin/sh 

echo `date "+DATE: %s"` >> ~/waiting_times.txt
curl -s "https://www.northwell.edu/wait-times?region=&zipcode=&default_view=list
&sort=shortest_wait&ed=true" | grep -A 7 wait-time__numeric | grep -v div >> ~/w
aiting_times.txt
