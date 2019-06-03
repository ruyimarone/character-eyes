echo "Testing all"
rm -r testing-en-01
#rm -r testing-ta-01
#rm -r testing-ta-02
echo "18.1%"
python model.py --dataset datasets/ud-2.3/en_ud23.pkl --log-dir testing-en-01 --dropout 0.5 --num-epochs 1 --log-to-stdout --debug |& tee repro.log
#python model.py --dataset datasets/ud-2.3/ta_ud23.pkl --log-dir testing-ta-01 --dropout 0.5 --num-epochs 1 --forward-dim 33 --backward-dim 95 --log-to-stdout --use-char-rnn |& tee -a repro.log
#python model.py --dataset datasets/ud-2.3/ta_ud23.pkl --log-dir testing-ta-02 --dropout 0.5 --num-epochs 1 --forward-dim 12 --backward-dim 4 --word-level-dim 16 --log-to-stdout --use-char-rnn |& tee -a repro.log

