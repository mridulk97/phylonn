import torch
import tqdm

EPS=1e-10

######### Misc

def getPredictions(logits):
    return torch.argmax(logits, dim=1)

######### Feature manipulation

import taming.constants as CONSTANTS

def accumulate_features(dataloader, model, output_keys=[CONSTANTS.QUANTIZED_PHYLO_OUTPUT], device=None):
    accumlated_features = {}
    accumulated_labels= None
    accumulated_predictions= None

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = batch['image'] 
        labels = batch['label']
        if device is not None:
            images = images.to(device)
        _, disentangler_loss_dic, _, in_out_disentangler = model(images)
        features = {}
        for output_key in output_keys:
            features_ = in_out_disentangler[output_key].detach().cpu()
            features_ = features.reshape(features.shape[0], -1)
            features[output_key] = features_
        pred = getPredictions(in_out_disentangler[CONSTANTS.DISENTANGLER_CLASS_OUTPUT])

        # Calculate distance for each pair.
        for output_key in output_keys:
            accumlated_features[output_key] = features[output_key] if accumlated_features[output_key] is None else torch.cat([accumlated_features[output_key], features[output_key]]).detach()
        accumulated_labels = labels.tolist() if accumulated_labels is None else accumulated_labels + labels.tolist()
        accumulated_predictions = pred.tolist() if accumulated_predictions is None else accumulated_predictions + pred.tolist()

    return accumlated_features, accumulated_labels, accumulated_predictions


def get_zqphylo_sub(zqphylo, num_phylo_levels =None, phylo_level=None, sub_vector_ratio=None):
    print(zqphylo.shape)
    zqphylo_size = zqphylo.shape[1]
    vector_unit_ration = 1/num_phylo_levels

    if sub_vector_ratio is not None:
        phylo_distance_cap = 1 - phylo_level
        sub_vector= slice(int((sub_vector_ratio-vector_unit_ration)*zqphylo_size), int(sub_vector_ratio*zqphylo_size))
    else:
        phylo_distance_cap = 0.
        sub_vector= slice(0, zqphylo_size)

    return phylo_distance_cap, sub_vector


def get_CosineDistance_matrix(features):
    ## TODO: we should replace the zq with cb indices really? Maybe Cos similarity is not the best measure?
    if features.dim() >2:
        features = features.reshape(features.shape[0], -1)

    features_norm = features / (EPS + features.norm(dim=1)[:, None])
    ans = torch.mm(features_norm, features_norm.transpose(0,1))

    # We want distance, not similarity.
    ans = torch.add(-ans, 1.)

    return ans

def get_HammingDistance_matrix(features):
    hamming_distance = torch.cdist(features.float(), features.float(), p=0)
    hamming_distance = hamming_distance/torch.max(hamming_distance)
    return hamming_distance

def get_species_pylo_distance(classnames, distance_function):
    ans = torch.zeros(len(classnames),len(classnames))

    for indx, i in enumerate(classnames):
        for indx2, j in enumerate(classnames[indx+1:]):
            dist = distance_function(i, j)
            # print(i, j, dist)
            ans[indx][indx2+1+indx] = ans[indx2+1+indx][indx] = dist
            
    distances_phylo_normalized = ans/(EPS+torch.max(ans))

    return distances_phylo_normalized


    

class Embedding_Code_converter():
    def __init__(self, get_codebook_entry_index_function, embedding_function, embedding_shape):
        self.embedding_function = embedding_function
        self.get_codebook_entry_index_function = get_codebook_entry_index_function
        self.embedding_shape = embedding_shape # (16, 8, 4))

    ### (n, 16, 8, 4) < - > (n, 32, 16)
    def reshape_zphylo(self, embedding, reverse = False):
        if not reverse:
            ans = embedding.reshape(embedding.shape[0], embedding.shape[1], -1).permute(0,2,1)
        else:
            ans = embedding.permute(0,2,1).reshape((embedding.shape[0], self.embedding_shape[0], self.embedding_shape[1], self.embedding_shape[2]))
        
        return ans

    # (n, 16, 8, 4) - > (n, 32)
    def get_phylo_codes(self, z_phylo, verify=False):
        embeddings = self.reshape_zphylo(z_phylo) 

        codes = torch.zeros((embeddings.shape[0], embeddings.shape[1])).to(z_phylo.device).long()
        for i in range(embeddings.shape[0]):
            for j in range(embeddings.shape[1]):
                codes[i, j] = self.get_codebook_entry_index_function(embeddings[i, j, :].reshape(1, -1))[0]

        if verify:
            embeddings = self.get_phylo_embeddings(codes, verify=False)
            assert torch.all(torch.isclose(embeddings, z_phylo))

        return codes

    # (n, 16, 8, 4) < - (n, 32)
    def get_phylo_embeddings(self, phylo_code, verify=False):
        embeddings = torch.zeros(self.embedding_shape).reshape(self.embedding_shape[0], -1).unsqueeze(0).repeat(phylo_code.shape[0], 1, 1).permute(0,2,1).to(phylo_code.device) # (n,32,16)
        for i in range(embeddings.shape[0]):
            for j in range(embeddings.shape[1]):
                embeddings[i, j, :] = self.embedding_function(phylo_code[i, j])

        embeddings = self.reshape_zphylo(embeddings, reverse=True)

        if verify:
            codes = self.get_phylo_codes(embeddings, verify=False)
            assert torch.all(torch.isclose(phylo_code, codes))

        return embeddings
    

