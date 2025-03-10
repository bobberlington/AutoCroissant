LOCAL_DIR_LOC   = "~/Desktop/TTSCardMaker"
ALIAS_PKL       = "aliases.pkl"
STATS_PKL       = "stats.pkl"
OLD_STATS_PKL   = "old_stats.pkl"

# Grodus, Bobberlington, Tyler, Croissant, Kyle, Jason, Ech, Zoey
bot_admin_ids = [247149441734279169, 80731173726191616, 346841750570270721, 451703959720296448, 264286438504267776, 213046347903795200, 381907548162359309, 190065758267637760]

# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py
sd15_rgb_factors = [
                    #   R        G        B
                    [ 0.3512,  0.2297,  0.3227],
                    [ 0.3250,  0.4974,  0.2350],
                    [-0.2829,  0.1762,  0.2721],
                    [-0.2120, -0.2616, -0.7177]
                ]

sdxl_rgb_factors = [
                    #   R        G        B
                    [ 0.3651,  0.4232,  0.4341],
                    [-0.2533, -0.0042,  0.1068],
                    [ 0.1076,  0.1111, -0.0362],
                    [-0.3165, -0.2492, -0.2188]
                ]
sdxl_rgb_factors_bias = [0.1084, -0.0175, -0.0011]

flux_rgb_factors = [
            [-0.00865,  0.0061,     0.017025],
            [-0.00865,  0.0061,     0.017025],
            [-0.00865,  0.0061,     0.017025],
            [-0.00865,  0.0061,     0.017025],
            [ 0.00085,  0.00525,    0.017175],
            [ 0.00085,  0.00525,    0.017175],
            [ 0.00085,  0.00525,    0.017175],
            [ 0.00085,  0.00525,    0.017175],
            [ 0.006875, -0.0167,   -0.010825],
            [ 0.006875, -0.0167,   -0.010825],
            [ 0.006875, -0.0167,   -0.010825],
            [ 0.006875, -0.0167,   -0.010825],
            [-0.00435,  0.004,      0.015425],
            [-0.00435,  0.004,      0.015425],
            [-0.00435,  0.004,      0.015425],
            [-0.00435,  0.004,      0.015425],
            [ 0.021475, 0.018025,   0.008225],
            [ 0.021475, 0.018025,   0.008225],
            [ 0.021475, 0.018025,   0.008225],
            [ 0.021475, 0.018025,   0.008225],
            [ 0.0001,   0.009575,   0.002875],
            [ 0.0001,   0.009575,   0.002875],
            [ 0.0001,   0.009575,   0.002875],
            [ 0.0001,   0.009575,   0.002875],
            [ 0.010125, 0.021525,   0.022875],
            [ 0.010125, 0.021525,   0.022875],
            [ 0.010125, 0.021525,   0.022875],
            [ 0.010125, 0.021525,   0.022875],
            [-0.0059,   -0.004625, -0.006475],
            [-0.0059,   -0.004625, -0.006475],
            [-0.0059,   -0.004625, -0.006475],
            [-0.0059,   -0.004625, -0.006475],
            [-0.006125, 0.00625,      0.0295],
            [-0.006125, 0.00625,      0.0295],
            [-0.006125, 0.00625,      0.0295],
            [-0.006125, 0.00625,      0.0295],
            [ 0.0252,   0.018875,  -0.010525],
            [ 0.0252,   0.018875,  -0.010525],
            [ 0.0252,   0.018875,  -0.010525],
            [ 0.0252,   0.018875,  -0.010525],
            [-0.012875, 0.005025,   0.000275],
            [-0.012875, 0.005025,   0.000275],
            [-0.012875, 0.005025,   0.000275],
            [-0.012875, 0.005025,   0.000275],
            [ 0.0107,   -0.0003,     -0.0009],
            [ 0.0107,   -0.0003,     -0.0009],
            [ 0.0107,   -0.0003,     -0.0009],
            [ 0.0107,   -0.0003,     -0.0009],
            [ 0.020425, 0.019125,   0.018725],
            [ 0.020425, 0.019125,   0.018725],
            [ 0.020425, 0.019125,   0.018725],
            [ 0.020425, 0.019125,   0.018725],
            [-0.0316,   -0.01305,  -0.027575],
            [-0.0316,   -0.01305,  -0.027575],
            [-0.0316,   -0.01305,  -0.027575],
            [-0.0316,   -0.01305,  -0.027575],
            [-0.007,    -0.022025, -0.012475],
            [-0.007,    -0.022025, -0.012475],
            [-0.007,    -0.022025, -0.012475],
            [-0.007,    -0.022025, -0.012475],
            [-0.03155,  -0.02455,   -0.01945],
            [-0.03155,  -0.02455,   -0.01945],
            [-0.03155,  -0.02455,   -0.01945],
            [-0.03155,  -0.02455,   -0.01945]
]

flux_rgb_factors_bias = [-0.0329, -0.0718, -0.0851]
