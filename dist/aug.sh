#!/bin/bash

python -m hwars.jobs.play '
{
  "repeat": 300,
  "commands":[
    {"button":"Start"},
    {"button":"To battle", "sleep": 3},
    {"key":"a"},
    {"button":"OK", "timeout": 600}
  ]
}'
