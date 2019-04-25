import json
import sys

from allennlp.commands import main

output_file = 'data/predictions/simple_fusion_elmo.json'
cuda_device = '-1'
predictor = 'snlive_fusion_predictor'
archive_file = 'models/simple_fusion_elmo/model.tar.gz'
input_file = 'data/snli_ve_test.jsonl'
#overrides = json.dumps({"trainer": {"cuda_device": -1}})

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "--output-file", output_file,
    "--silent",
    "--cuda-device", cuda_device,
    "--predictor", predictor,
    "--include-package", "snli_ve",
    archive_file,
    input_file
]

main()
