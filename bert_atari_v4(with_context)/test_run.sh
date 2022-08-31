# Decision Transformer (DT) reward_conditioned GoToObjMazeS4 GoToObjMazeS5 GoToObjMazeS6 GoToObjMazeS7  GoToObjMazeS4R2 GoToLocalS7N4 GoToLocalS7N5 GoToLocalS8N3 GoToLocalS8N5 GoToLocalS8N7
# GoToObjMazeS6 GoToObjMazeS7
#for env in GoToObjMazeS4 GoToObjMazeS5 
#do
#    for seed in 123 231 321 132 213
#    do
#        python run_dt_atari.py --seed $seed --horizon 10 --context_length 10 --sample_iteration 50 --epochs 200 --env $env --model_type 'naive' --batch_size 64
#    done
#done 231 321 132 213 GoToObjMazeS4Close PutNextLocal GoToSeqS5R2

for seed in 123
do
    python run_dt_atari.py --seed $seed --horizon 5 --context_length 25 --sample_iteration 10 --epochs 200 --model_type 'naive' --num_steps 10000 --num_buffers 5 --game 'Breakout' --batch_size 64
done
