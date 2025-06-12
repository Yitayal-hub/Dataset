#!/bin/bash

python3 get_char_vocab.py

python3 filter_embeddings.py cc.am.300.vec train.amharic.jsonlines dev.amharic.jsonlines test.amharic.jsonlines

