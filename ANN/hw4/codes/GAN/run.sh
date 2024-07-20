for LD in 16 32 64 100
do
    for HD in 16 32 64 100
    do
        python main.py --do_train --latent_dim $LD --generator_hidden_dim $HD --discriminator_hidden_dim $HD
    done
done