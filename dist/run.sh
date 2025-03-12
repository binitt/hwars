#!/bin/bash

python -m hwars.jobs.play '
{
  "repeat": 300,
  "commands":[
    {"button":"Find", "index":-3},
    {"button":"Start"},
	{"button": "Cross", "conditions": ["0.99 USD"], "optional": true, "back": true, "timeout": 5},
    {"button":"To battle"},
    {"button":"Return to", "timeout": 300}
  ]
}'

