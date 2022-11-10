from taming.modules.losses.phyloloss import Species_sibling_finder, parse_phyloDistances, get_relative_distance_for_level
from taming.data.phylogeny import Phylogeny

def main():
    phyloDistances_string = "0.77,0.5,0.33"
    p = Phylogeny("/home/elhamod/data/Fish")
    d = parse_phyloDistances(phyloDistances_string)
    s = Species_sibling_finder(p, d)
    print('*******')
    
    
    for k, k2 in enumerate(d):
        d_0 = get_relative_distance_for_level(d, k)
        print('phyloDistances_string',phyloDistances_string)
        print('parsed distances',d)
        print('distance furthest from leaves',d_0)
        print('*******')
        print('*******')
        print('*******')
        
        g = list(p.get_species_groups(d_0))
        for u, u2 in enumerate(g):
            group_0 = g[u]
            
            for q, q2 in enumerate(group_0):
                print('$$$$$', k2, u2, q2)
                
                species_0_0 = group_0[q]
                print('group', group_0, len(group_0))
                print('*******')
                
                sib1 = p.get_siblings_by_name(species_0_0, d_0)
                print('siblings', sib1, len(sib1))
                print('*******')
                
                loss_name = str(d[k]).replace(".", "")+"distance"
                print(loss_name)
                sib2 = s.map_speciesId_siblingVector(p.getLabelList().index(species_0_0), loss_name)
                sib2 = list(map(lambda x: p.getLabelList()[x], sib2))
                print('siblings', sib2, len(sib2))
                print('*******')
                print('*******')
                print('*******')
                
                assert len(sib2) == len(sib1)
                assert len(group_0) == len(sib1)
    

if __name__ == "__main__":
    main()
    
    
