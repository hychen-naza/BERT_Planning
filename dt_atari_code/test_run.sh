# Decision Transformer (DT)
for seed in 123
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 10 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 64
done
