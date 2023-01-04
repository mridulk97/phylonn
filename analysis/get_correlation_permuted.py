import os
import pandas as pd
import argparse
from omegaconf import OmegaConf
import numpy as np
from taming.plotting_utils import dump_to_json
from scipy import stats
import statistics 

NUM_SAMPLES=10
##########

def main(configs_yaml):
    path = configs_yaml.path    
    output_names = configs_yaml.output_names
    arrays1 = configs_yaml.arrays1
    arrays2 = configs_yaml.arrays2
    file_name = configs_yaml.file_name
    
    distances = {}
    for indx, name in enumerate(output_names):
        distances[name] = []
        for i in range(NUM_SAMPLES):
            p1 = os.path.join(path, arrays1[indx])
            p2 = os.path.join(arrays2[indx])
            
            df1 = pd.read_csv(p1)
            df1 = df1.sample(frac=1) 
            df1 = df1.to_numpy()
            df2 = pd.read_csv(p2).to_numpy()
            
            df1 = df1[np.triu_indices(df1.shape[0], k = 1)]
            df2 = df2[np.triu_indices(df2.shape[0], k = 1)]
            
            # distance = np.corrcoef(df1,df2)[0, 1]
            distance = stats.spearmanr(df1, df2).correlation
            # distance  = ((df1 - df2) ** 2).mean() ** 0.5
            
            distances[name] = distances[name] + [distance]
        print(name, 'mean', statistics.mean(distances[name]), 'std', statistics.stdev(distances[name]))
    
    dump_to_json(distances, path, name=file_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/get_correlation.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    # cfg = parser.config
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)