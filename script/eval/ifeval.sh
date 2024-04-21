
MODEL=$1
defense=$2

if [ -z "$defense" ]; then
    defense=""
    echo "No defense specified"
fi

python -m instruction_following_eval.generate_main \
    --model $MODEL \
    --defense $defense
