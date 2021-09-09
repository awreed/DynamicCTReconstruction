from dynamic_ct_reconstruction_pipeline import DynamicCTReconstructionPipeline
import sys

if __name__ == '__main__':
    config_file = sys.argv[1]
    print("Running pipeline with config file", config_file)

    DCTRP = DynamicCTReconstructionPipeline(config_file)

    DCTRP.run_pipeline()
