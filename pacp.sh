#!/bin/bash
eval $(ssh-agent -s);
ssh-add ~/.ssh/devel2-tfraw;
git status;
git pull && git add -A && git commit -a -m Msg && git push;
sudo pkill -f ssh-agent;

#eof