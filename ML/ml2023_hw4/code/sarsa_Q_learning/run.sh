for algorithm in Sarsa Q_learning
do
    for lr in 0.01 0.1 1 2
    do
        python main.py --lr $lr --algorithm $algorithm > log/$algorithm\_$lr.log
    done
done