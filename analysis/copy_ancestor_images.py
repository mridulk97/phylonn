import shutil
import os


RootDir1 = '/home/mridul/taming-transformers/logs/2022-11-09T15-00-05_CW-VQGAN-transformer256img-cw-transformer/figs/transformer_generated_dataset/species_top_5'

TargetFolder = r'*your target folder*'
level2_folders = [['Alosa chrysochloris'],
['Carassius auratus', 'Cyprinus carpio'] ,
['Esox americanus'],
['Gambusia affinis'],
['Lepisosteus osseus', 'Lepisosteus platostomus'],
['Lepomis auritus', 'Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus', 'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis', 'Lepomis microlophus'],
['Morone chrysops', 'Morone mississippiensis'],
['Notropis atherinoides', 'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 'Notropis stramineus', 'Notropis telescopus', 'Notropis texanus', 'Notropis volucellus', 'Notropis wickliffi', 'Phenacobius mirabilis'],
['Noturus exilis', 'Noturus flavus', 'Noturus gyrinus', 'Noturus miurus', 'Noturus nocturnus']]

for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
        for name in files:
            if name.endswith('.csv'):
                print("Found")
                SourceFolder = os.path.join(root,name)
                shutil.copy2(SourceFolder, TargetFolder)