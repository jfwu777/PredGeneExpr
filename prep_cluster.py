import sys
import pandas as pd
import re

seq_id_patt = re.compile("(train|test)_([0-9]+)_score_([0-9]+\.?[0-9]*)")

def name_to_index(seq: str):
    m = seq_id_patt.match(seq)
    if m:
        return int(m.groups()[1])
    else:
        return None


if __name__ == "__main__":
    df = pd.read_csv(
        sys.argv[1], sep='\t', header=None, names=["sequence", "score"]
    )
    cluster = pd.read_csv(
        sys.argv[2], sep='\t', header=None, names=["cluster", "sequence"]
    )
    cluster["cluster_id"] = cluster["cluster"].apply(name_to_index)
    cluster.index = cluster["sequence"].apply(name_to_index)
    df_new = df.join(cluster[["cluster_id"]])
    df_new.to_csv(sys.stdout, sep="\t", index=False)