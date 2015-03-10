#!/bin/bash

# INPUT: from stdin

# - ignore lines containing digits
# - lowercase all letters
# - remove characters besides a-z, :;.?!(), comma, space (NOTE: WE REMOVE \n!!!)
# - squash extra spaces together

grep -v '[0-9]' | tr '[:upper:]\n' '[:lower:] ' | tr -d -c '[:digit:][:lower:]:;.?!)(, ' | tr -s " "

