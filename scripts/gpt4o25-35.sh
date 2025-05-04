DATA_ROOT=../datasets
# outdir=${DATA_ROOT}/exprs_map/test/
outdir=./scripts/log_4o_enhanced_input_57cases

# flag="--root_dir ${DATA_ROOT}
#       --img_root ../datasets/RGB_Observations_nochange
#       --split MapGPT_72_scenes_processed
#       --end 20  # the number of cases to be tested
#       --output_dir ${outdir}
#       --max_action_len 15
#       --save_pred
#       --stop_after 3
#       --llm gpt-4o-2024-05-13
#       --response_format json
#       --max_tokens 1000
#       "

flag="--root_dir ${DATA_ROOT}
      --img_root ../datasets/RGB_Observations_nochange
      --split MapGPT_72_scenes_processed
      --start 25
      --end 35
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm gpt-4o
      --response_format json
      --max_tokens 1000
      "

python vln/main_gpt.py $flag
