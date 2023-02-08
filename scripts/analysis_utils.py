from scripts.modules.losses.phyloloss import Species_sibling_finder, get_loss_name, get_relative_distance_for_level, parse_phyloDistances

import torch
from scipy.spatial import distance as js_distance
import pickle

EPS=1e-10

####### Level CrossEntropy 

def get_phylomapper_from_config(phylogeny, phyloDistances_string, level):
    phylo_distances = parse_phyloDistances(phyloDistances_string)
    siblingfinder = Species_sibling_finder(phylogeny, phylo_distances)
    
    relative_distance = get_relative_distance_for_level(phylo_distances, level)
    species_groups = phylogeny.get_species_groups(relative_distance)
    species_groups_representatives = list(map(lambda x: x[0], species_groups))
    mlb = list(map(lambda x: phylogeny.getLabelList().index(x), species_groups_representatives))
    
    outputname = get_loss_name(phylo_distances, level)
    
    return PhylogenyMapper(level, siblingfinder.map_speciesId_siblingVector, mlb, outputname)
    


class PhylogenyMapper:
    def __init__(self, level, sibling_mapper, mlb, outputname):
        self.level = level
        
        self.sibling_mapper = sibling_mapper
        self.mlb = mlb
        
        self.outputname = outputname
        
    def get_len(self):
        return len(self.mlb)
    
    # maps from fresh level indexing to species indexing
    def get_reverse_indexing(self, truth):
        return list(map(lambda x: self.mlb[x], truth))
    
    # maps from species indexing to level indexing by species indexing convention
    def get_original_indexing_truth(self, truth):
        return list(map(lambda x: self.sibling_mapper(x, self.outputname)[0], truth))
    
    # maps from species indexing to fresh level indexing
    def get_mapped_truth(self, truth):
        sibling_truth = self.get_original_indexing_truth(truth)
        return torch.LongTensor(list(map(lambda x: self.mlb.index(x), sibling_truth))).to(truth.device)


######### Histogram misc

def js_divergence(A, B):
    return js_distance.jensenshannon(A,B)


class HistogramParser:
    def __init__(self, model):
        self.possible_codes = model.phylo_disentangler.n_embed
        self.n_phylolevels = model.phylo_disentangler.n_phylolevels
        self.codebooks_per_phylolevel = model.phylo_disentangler.codebooks_per_phylolevel
        self.n_levels_non_attribute = model.phylo_disentangler.n_levels_non_attribute
    
    def get_distances(self, hist, species1, species2):
        hist_species1 = hist[species1]
        hist_species2 = hist[species2]
        
        distances = []
        most_common1 = []
        most_common2 = []
        
        for location_code in range(len(hist_species1)):
            hist_species1_location = hist_species1[location_code]
            hist_species2_location = hist_species2[location_code]
            
            hist_species1_location_histogram = torch.histc(torch.Tensor(hist_species1_location), bins=self.possible_codes, min=0, max=self.possible_codes-1)
            hist_species1_location_histogram = hist_species1_location_histogram/torch.sum(hist_species1_location_histogram)
            most_common1.append(torch.argmax(hist_species1_location_histogram))
            
            hist_species2_location_histogram = torch.histc(torch.Tensor(hist_species2_location), bins=self.possible_codes, min=0, max=self.possible_codes-1)
            hist_species2_location_histogram = hist_species2_location_histogram/torch.sum(hist_species2_location_histogram)
            most_common2.append(torch.argmax(hist_species2_location_histogram))
        
        
            d = js_divergence(hist_species1_location_histogram, hist_species2_location_histogram)
            distances.append(d)
    
        distances = torch.tensor(distances)
        most_common1 = torch.tensor(most_common1)
        most_common2 = torch.tensor(most_common2)
        
        return distances, most_common1, most_common2


######## Histogram freq construction

class HistogramFrequency:
    def __init__(self, num_of_classes, num_of_locations, num_of_locations_nonattr=None):
        self.hist_arr = [[] for x in range(num_of_classes)]
        for i in range(len(self.hist_arr)):
            self.hist_arr[i] = [[] for x in range(num_of_locations)]
        
        self.hist_arr_nonattr = None 
        if num_of_locations_nonattr is not None:
            self.hist_arr_nonattr = [[] for x in range(num_of_classes)]
            for i in range(len(self.hist_arr)):
                self.hist_arr_nonattr[i] = [[] for x in range(num_of_locations_nonattr)]
            
    def load_from_file(self, file_path):
        temp = pickle.load(open(file_path, "rb"))
        self.hist_arr=temp[0]
        self.hist_arr_nonattr=temp[1]
        print(file_path, 'loaded!')
        
    def save_to_file(self, file_path):
        pickle.dump((self.hist_arr, self.hist_arr_nonattr), open(file_path, "wb"))
        print(file_path, 'saved!')
    
    def set_location_frequencies(self, lbl,output_indices, output_indices_nonattr=None):
        # iterate through code locations
        for code_location in range(output_indices.shape[1]):
            code = output_indices[0, code_location]
            self.hist_arr[lbl][code_location].append(code.item()) 
            
        if self.hist_arr_nonattr is not None:
            for code_location in range(output_indices_nonattr.shape[1]):
                code = output_indices_nonattr[0, code_location]
                self.hist_arr_nonattr[lbl][code_location].append(code.item()) 

######### Misc

def getPredictions(logits):
    return torch.argmax(logits, dim=1)

######### Feature manipulation




def aggregate_metric_from_specimen_to_species(sorted_class_names_according_to_class_indx, specimen_distance_matrix):
    unique_sorted_class_names_according_to_class_indx = sorted(set(sorted_class_names_according_to_class_indx))

    species_dist_matrix = torch.zeros(len(unique_sorted_class_names_according_to_class_indx), len(unique_sorted_class_names_according_to_class_indx))
    for indx_i, i in enumerate(unique_sorted_class_names_according_to_class_indx):
        class_i_indices = [idx for idx, element in enumerate(sorted_class_names_according_to_class_indx) if element == i] 
        for indx_j, j in enumerate(unique_sorted_class_names_according_to_class_indx[indx_i:]):
            class_j_indices = [idx for idx, element in enumerate(sorted_class_names_according_to_class_indx) if element == j] 
            i_j_mean_embeddign_distance = torch.mean(specimen_distance_matrix[class_i_indices, :][:, class_j_indices])
            species_dist_matrix[indx_i][indx_j+ indx_i] = species_dist_matrix[indx_j+ indx_i][indx_i] = i_j_mean_embeddign_distance
            
    return species_dist_matrix

    
def get_CosineDistance_matrix(features):
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


class Embedding_Code_converter():
    def __init__(self, get_codebook_entry_index_function, embedding_function, embedding_shape):
        self.embedding_function = embedding_function
        self.get_codebook_entry_index_function = get_codebook_entry_index_function
        self.embedding_shape = embedding_shape # (16, 8, 4))

    
    # k(32) <-> k(8),j(4)
    def get_code_reshaped_index(self, i, j=None):
        n_levels = self.embedding_shape[-1]

        if j is not None:
            return i*n_levels + j
        else:
            return i//n_levels, i%n_levels, 

    def get_sub_level(self, code, level):
        code_reshaped = self.reshape_code(code, reverse = True)
        sub_code_reshaped = code_reshaped[:,:,:level+1]
        sub_code_reshaped_back = self.reshape_code(sub_code_reshaped)
        return sub_code_reshaped_back
    
    def get_level(self, code, level):
        code_reshaped = self.reshape_code(code, reverse = True)
        sub_code_reshaped = code_reshaped[:,:,level].reshape(code_reshaped.shape[0], code_reshaped.shape[1], 1)
        sub_code_reshaped_back = self.reshape_code(sub_code_reshaped)
        return sub_code_reshaped_back
    
    def get_post_level(self, code, level):
        code_reshaped = self.reshape_code(code, reverse = True)
        sub_code_reshaped = code_reshaped[:,:,level:]
        sub_code_reshaped_back = self.reshape_code(sub_code_reshaped)
        return sub_code_reshaped_back
        
    def set_sub_level(self, code, replacement, index):
        code_reshaped = self.reshape_code(code, reverse = True)
        replacement_reshaped = self.reshape_code(replacement, reverse = True)
        answer =torch.clone(code_reshaped)
        answer[:, :, :index+1] = replacement_reshaped
        answer_reshaped = self.reshape_code(answer)
        return answer_reshaped
    
    
    # k, 32 -> (k, 8, :level),(k, 8, level:) -> corresponding codes.
    def split_codes(self, code, level, n_phylolevel):
        code_postfix = self.get_post_level(code, level)
        code_prefix = self.get_sub_level(code, level-1)
        assert code.shape[-1] == code_postfix.shape[-1] + code_prefix.shape[-1]
        return code_prefix, code_postfix
    
    # codes prefix and postfdix -> combined.
    def merge_codes(self, code_prefix, code_postfix, n_phylolevels):
        code_postfix_reshaped = self.reshape_code(code_postfix, reverse = True)
        code_prefix_reshaped = self.reshape_code(code_prefix, reverse = True)
        code_reshaped = torch.cat([code_prefix_reshaped,code_postfix_reshaped ], dim=-1)
        assert code_reshaped.shape[-1] == n_phylolevels
        return self.reshape_code(code_reshaped)
        
        
        
        

    
    ### (n, 8, k) < - > (n, 8*k)
    def reshape_code(self, code, reverse = False):
        if not reverse:
            ans = code.reshape(code.shape[0], -1)
        else:
            ans = code.reshape((code.shape[0], self.embedding_shape[-2], -1))
        
        return ans
    
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
    

