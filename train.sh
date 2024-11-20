SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"
curl -X POST -H 'Content-type: application/json' --data '{"text":"Training start"}' $SLACK_WEBHOOK_URL

python train.py --goal imagenet --source dtd --data_path ~/Datasets \
 --scale small --target_width 64 --target_height 64 \
 --wmin 0.1 --wmax 1.0 --hmin 0.1 --hmax 1.0 \
 --algo sac --model_based --noop \
 --replay_size 80000 --num_steps 10 --automatic_entropy_tuning \
 --num_episodes 50000 --mse_reward

curl -X POST -H 'Content-type: application/json' --data '{"text":"Training with dis done."}' $SLACK_WEBHOOK_URL

python train.py --goal imagenet --source dtd --data_path ~/Datasets \
 --scale small --target_width 64 --target_height 64 \
 --wmin 0.1 --wmax 1.0 --hmin 0.1 --hmax 1.0 \
 --algo sac --model_based --noop \
 --replay_size 80000 --num_steps 10 --automatic_entropy_tuning \
 --num_episodes 50000

curl -X POST -H 'Content-type: application/json' --data '{"text":"Training done."}' $SLACK_WEBHOOK_URL