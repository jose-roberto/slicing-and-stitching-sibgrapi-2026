class Hyperparameters:

    def __init__(self):
    
        super().__init__()
        
    def get_hyperparameters(self, tag):

        if tag == "cnn_structseg_thorax":

            hps_cnn_structseg_thorax = {
                'both_lungs': {         
                    'base_lr': 1e-4,
                    'weight_decay': 1e-5, 
                    'max_lr': 4e-3,     
                    'pct_start': 0.10,   
                    'div_factor': 40,
                    'final_div_factor': 1e3
                },
                'esophagus': {          
                    'base_lr': 1e-5,    
                    'weight_decay': 1e-4, 
                    'max_lr': 8e-4,     
                    'pct_start': 0.35,   
                    'div_factor': 20,
                    'final_div_factor': 1e5
                },
                'heart': {
                    'base_lr': 5e-5,
                    'weight_decay': 3e-5,
                    'max_lr': 2e-3,      
                    'pct_start': 0.20,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                'spinal_cord': {
                    'base_lr': 2e-5,
                    'weight_decay': 1e-4, 
                    'max_lr': 1e-3,
                    'pct_start': 0.30,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                'trachea': {
                    'base_lr': 8e-5,
                    'weight_decay': 2e-5,
                    'max_lr': 3e-3,      
                    'pct_start': 0.15,
                    'div_factor': 30,
                    'final_div_factor': 1e3
                }
            }
            
            return hps_cnn_structseg_thorax
            
        elif tag == "cnn_multiorgan_ct_btcv":
            
            hps_cnn_multiorgan_ct_btcv = {
                'both_adrenal_glands': { 
                    'base_lr': 1e-4,        
                    'weight_decay': 1e-5,  
                    'max_lr': 3e-3,         
                    'pct_start': 0.10,      
                    'div_factor': 30,
                    'final_div_factor': 1e4 
                },
                'both_kidneys': {
                    'base_lr': 5e-5,
                    'weight_decay': 3e-5,
                    'max_lr': 2e-3,      
                    'pct_start': 0.20,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                'duodenum': {
                    'base_lr': 2e-5,
                    'weight_decay': 8e-5,
                    'max_lr': 1e-3,      
                    'pct_start': 0.30,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                'esophagus': {          
                    'base_lr': 1e-4,
                    'weight_decay': 1e-5, 
                    'max_lr': 4e-3,     
                    'pct_start': 0.15,   
                    'div_factor': 40,
                    'final_div_factor': 1e3
                },
                'gallbladder': { 
                    'base_lr': 5e-5,
                    'weight_decay': 1e-5,   
                    'max_lr': 2e-3,         
                    'pct_start': 0.15,      
                    'div_factor': 40,
                    'final_div_factor': 1e4
                },
                'liver': {         
                    'base_lr': 1e-4,
                    'weight_decay': 1e-5, 
                    'max_lr': 4e-3,     
                    'pct_start': 0.15,   
                    'div_factor': 40,
                    'final_div_factor': 1e3
                },
                'major_arteries_and_veins': {        
                    'base_lr': 1e-4,
                    'weight_decay': 1e-5, 
                    'max_lr': 4e-3,     
                    'pct_start': 0.15,   
                    'div_factor': 40,
                    'final_div_factor': 1e3
                },
                'pancreas': {      
                    'base_lr': 2e-5,    
                    'weight_decay': 1e-4,   
                    'max_lr': 1e-3,         
                    'pct_start': 0.25,     
                    'div_factor': 25,
                    'final_div_factor': 1e4 
                },
                'spleen': {         
                    'base_lr': 1e-4,
                    'weight_decay': 1e-5, 
                    'max_lr': 3e-3,     
                    'pct_start': 0.15,   
                    'div_factor': 30,
                    'final_div_factor': 1e3
                },
                'stomach': {
                    'base_lr': 2e-5,
                    'weight_decay': 8e-5,
                    'max_lr': 1e-3,      
                    'pct_start': 0.30,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                }
            }
            
            return hps_cnn_multiorgan_ct_btcv
            
        elif tag == "att_structseg_thorax":

            hps_unetr_structseg_thorax = {
                'both_lungs': {         
                    'base_lr': 5e-5,
                    'weight_decay': 3e-5,
                    'max_lr': 8e-4,      
                    'pct_start': 0.15,
                    'div_factor': 20,
                    'final_div_factor': 1e3
                },
                 'esophagus': {         
                    'base_lr': 1e-5,
                    'weight_decay': 5e-4, 
                    'max_lr': 3e-4, 
                    'pct_start': 0.35,   
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                'heart': {
                    'base_lr': 2e-5,
                    'weight_decay': 1e-4, 
                    'max_lr': 4e-4,      
                    'pct_start': 0.25,
                    'div_factor': 20,
                    'final_div_factor': 1e4
                },
                'spinal_cord': {
                    'base_lr': 1e-5,
                    'weight_decay': 2e-4,
                    'max_lr': 2.5e-4,
                    'pct_start': 0.30,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                'trachea': {
                    'base_lr': 3e-5,
                    'weight_decay': 5e-5,
                    'max_lr': 5e-4,      
                    'pct_start': 0.20,
                    'div_factor': 15,
                    'final_div_factor': 1e3
                }
            }

            return hps_unetr_structseg_thorax
            
        elif tag == "att_multiorgan_ct_btcv":

            hps_unetr_multiorgan_ct_btcv = {
                'both_adrenal_glands': {
                    'base_lr': 1e-5,        
                    'weight_decay': 1e-4,   
                    'max_lr': 3e-4,        
                    'pct_start': 0.25,     
                    'div_factor': 30,
                    'final_div_factor': 1e4 
                },
                'both_kidneys': {
                    'base_lr': 5e-6,
                    'weight_decay': 1e-4,
                    'max_lr': 2e-4,      
                    'pct_start': 0.20,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                'duodenum': {
                    'base_lr': 5e-6,
                    'weight_decay': 2e-4,
                    'max_lr': 1e-4,      
                    'pct_start': 0.30,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                },
                'esophagus': {              
                    'base_lr': 1e-5,
                    'weight_decay': 1e-4, 
                    'max_lr': 4e-4,     
                    'pct_start': 0.20,   
                    'div_factor': 40,
                    'final_div_factor': 1e3
                },
                'gallbladder': {
                    'base_lr': 5e-6,
                    'weight_decay': 1e-4,   
                    'max_lr': 2e-4,         
                    'pct_start': 0.20,      
                    'div_factor': 40,
                    'final_div_factor': 1e4
                },
                'liver': {         
                    'base_lr': 1e-5,
                    'weight_decay': 1e-4, 
                    'max_lr': 4e-4,        
                    'pct_start': 0.15,   
                    'div_factor': 40,
                    'final_div_factor': 1e3
                },
                'major_arteries_and_veins': { 
                    'base_lr': 1e-5,
                    'weight_decay': 2e-4,   
                    'max_lr': 4e-4,     
                    'pct_start': 0.20,   
                    'div_factor': 40,
                    'final_div_factor': 1e3
                },
                'pancreas': {          
                    'base_lr': 5e-6,    
                    'weight_decay': 3e-4,  
                    'max_lr': 1e-4,         
                    'pct_start': 0.30,     
                    'div_factor': 20,
                    'final_div_factor': 1e4 
                },
                'spleen': {         
                    'base_lr': 1e-5,
                    'weight_decay': 1e-4, 
                    'max_lr': 3e-4,     
                    'pct_start': 0.15,   
                    'div_factor': 30,
                    'final_div_factor': 1e3
                },
                'stomach': {
                    'base_lr': 5e-6,
                    'weight_decay': 2e-4,
                    'max_lr': 1e-4,      
                    'pct_start': 0.30,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                }
            }
            
            return hps_unetr_multiorgan_ct_btcv
            
        else:
            raise ValueError(f'Unknown tag: {tag}')
