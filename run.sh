# Set up

conda create -n faỉrseq python=3.8 -y

conda activate fairseq

pip install --editable ./
pip install pyarrow

# update-moreh --force --torch 1.13.1

python setup.py build_ext --inplace


# Preprocess data

cd examples/translation/
bash prepare-iwslt14.sh
cd ../..
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir data-bin/iwslt14.tokenized.de-en


# Train

mkdir -p checkpoints/fconv

MOREH_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
    --optimizer nag --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir checkpoints/fconv  2>&1 | tee log.txt