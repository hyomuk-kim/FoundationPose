ps aux | grep ros | grep root | awk '{print $2}' | xargs kill -9
