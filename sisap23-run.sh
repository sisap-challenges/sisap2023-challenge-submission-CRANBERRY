#! /bin/bash
conda run -n CRANBERRY python search.py --size $1 --k 10 --buildindex True
