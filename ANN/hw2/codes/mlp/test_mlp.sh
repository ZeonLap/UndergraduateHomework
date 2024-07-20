for dr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
python main.py --drop_rate $dr --name mlp_dr_$dr
done