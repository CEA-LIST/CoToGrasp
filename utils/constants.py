import torch
import os

ROOT_PATH = os.environ['PYTHON_HOME_PATH'] + 'cotograsp'
DATA_PATH = os.environ['PYTHON_DATA_PATH'] + 'COTOGRASP_DATA'


###############################################################################################################################
### ALLEGRO 3+6+16                                                                                                          ###
###############################################################################################################################

ALLEGRO_CANONICAL_HAND_POSE = torch.tensor([0.0, 0.0, 0.0] +  # translation
                                            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] +  # uv rotation
                                            [0.0, 0.3, 0.3, 0.3] +              # Little
                                            [0.0, 0.3, 0.3, 0.3] +              # Middle
                                            [0.0, 0.3, 0.3, 0.3] +              # Index 
                                            [0.5, 0.5, 0.3, 0.3],               # Thumb
                                            dtype=torch.float)

ALLEGRO_FULL_STRAIGHT_POSE = torch.tensor([0.0, 0.0, 0.0] +  # translation
                                            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] +    # uv rotation
                                            [0.0, 0.0, 0.0, 0.0] +              # Little
                                            [0.0, 0.0, 0.0, 0.0] +              # Middle
                                            [0.0, 0.0, 0.0, 0.0] +              # Index
                                            [0.0, 1.1, 0.0, 0.0],               # Thumb
                                            dtype=torch.float)
ALLEGRO_OPTI_START_POSE = torch.tensor([0.0, 0.0, 0.0] +  # translation
                                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] +    # uv rotation
                                        [0.0, 0.0, 0.0, 0.0] +              # Little
                                        [0.0, 0.0, 0.0, 0.0] +              # Middle
                                        [0.0, 0.0, 0.0, 0.0] +              # Index
                                        [1.3, 0.0, 0.0, 0.0],               # Thumb
                                        dtype=torch.float)


ALLEGRO_GRASPS_LABELS = {
    'm1': [8, 10, 13, 15],
    'm2': [9, 11, 20],
    'm3': [10, 11, 20],
    'm4': [10, 11, 14, 15, 20],
    'm5': [10, 11, 14, 15, 17, 20],
#   'm6': Does not exist as allegro has only 3 fingers + 1 thumb
    'm7': [8, 9, 12, 13, 16, 19],
    'm8': [10, 11, 13, 15, 20],
    'm9': [4, 7, 10, 11, 13, 15, 20],
    'm10': [4, 7, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20],
    'm11': [1, 6, 8, 9, 12, 13, 14, 15, 17, 20],
    'm12': [8, 9, 10, 11, 18, 19, 20],
    'm13': [4, 7, 8, 9, 10, 11, 18, 19, 20],
    'm14': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'm15': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20],
    'm16': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20],
    'm17': [2, 10, 11, 14, 15, 17, 18, 19, 20],
    'm18': [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20],
    'm19': [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'm20': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'm21': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
}

# J: 8 9 10 11 4 5 6 7 0 1 2 3 12 13 14 15
ALLEGRO_JOINT_NAMES = [
    "LFJ0", "LFJ1", "LFJ2", "LFJ3",  # Little
    "MFJ0", "MFJ1", "MFJ2", "MFJ3",  # Middle
    "IFJ0", "IFJ1", "IFJ2", "IFJ3",  # Index
    "THJ0", "THJ1", "THJ2", "THJ3",  # Thumb
]

ALLEGRO_LINK_NAMES = {
    "little": ["link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"],
    "middle": ["link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"],
    "index": ["link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"],
    "thumb": ["link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"],
}


##################################################################################################################################
### SHADOWHAND 3+6+22                                                                                                          ###
##################################################################################################################################

SHADOWHAND_LINK_NAMES = {
    "index": ['ffproximal', 'ffmiddle', 'ffdistal'],
    "middle": ['mfproximal', 'mfmiddle', 'mfdistal'],
    "ring": ['rfproximal', 'rfmiddle', 'rfdistal'],
    "little": ['lfmetacarpal', 'lfproximal', 'lfmiddle', 'lfdistal'],
    "thumb": ['thproximal', 'thmiddle', 'thdistal'],
}


SHADOWHAND_CANONICAL_HAND_POSE = torch.tensor([0.0, 0.0, 0.0] +  # translation
                                            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] +  # uv rotation
                                            [0.0, 0.39, 0.39, 0.39] +              # INDEX
                                            [0.0, 0.39, 0.39, 0.39] +              # MIDDLE
                                            [0.0, 0.39, 0.39, 0.39] +              # RING
                                            [0.0, 0.0, 0.39, 0.39, 0.39] +         # LITTLE
                                            [-1.0, 0.33, 0.0, 0.0, 0.39],            # THUMB
                                            dtype=torch.float)

SHADOWHAND_FULL_STRAIGHT_POSE = torch.tensor([0.0, 0.0, 0.0] +                  # translation
                                            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] +    # uv rotation
                                            [0.0, 0.0, 0.0, 0.0] +              # INDEX
                                            [0.0, 0.0, 0.0, 0.0] +              # MIDDLE
                                            [0.0, 0.0, 0.0, 0.0] +              # RING
                                            [0.0, 0.0, 0.0, 0.0, 0.0] +         # LITTLE
                                            [-1.0, 0.0, 0.0, 0.0, 0.0],          # THUMB
                                            dtype=torch.float)

SHADOWHAND_OPTI_START_POSE = torch.tensor([0.0, 0.0, 0.0] +                  # translation
                                            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] +    # uv rotation
                                            [0.0, 0.0, 0.0, 0.0] +              # INDEX
                                            [0.0, 0.0, 0.0, 0.0] +              # MIDDLE
                                            [0.0, 0.0, 0.0, 0.0] +              # RING
                                            [0.0, 0.0, 0.0, 0.0, 0.0] +         # LITTLE
                                            [0.0, 1.2, 0.0, 0.0, 0.0],          # THUMB
                                            dtype=torch.float)

SHADOWHAND_FOREARM2WRIST = torch.tensor([0.0, -0.010, 0.21301])  # x y z
SHADOWHAND_WRIST2PALM = torch.tensor([0.0, 0.0, 0.034])  # x y z

SHADOWHAND_GRASPS_LABELS = {
    'm1': [8, 10, 13, 15],
    'm2': [9, 11, 22],
    'm3': [10, 11, 22],
    'm4': [10, 11, 14, 15, 22],
    'm5': [10, 11, 14, 15, 17, 22],
    'm6': [10, 11, 14, 15, 17, 19, 22],
    'm7': [8, 9, 12, 13, 16, 18, 21],
    'm8': [10, 11, 13, 15, 22],
    'm9': [4, 7, 10, 11, 13, 15, 22],
    'm10': [4, 7, 10, 11, 12, 13, 14, 15, 17, 20, 21, 22],
    'm11': [1, 6, 8, 9, 12, 13, 14, 15, 17, 19, 22],
    'm12': [8, 9, 10, 11, 20, 21, 22],
    'm13': [4, 7, 8, 9, 10, 11, 20, 21, 22],
    'm14': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'm15': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22],
    'm16': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22],
    'm17': [2, 10, 11, 14, 15, 17, 19, 20, 21, 22],
    'm18': [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22],
    'm19': [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22],
    'm20': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'm21': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
}



#################################################################################################################################
### COMMON CONSTANTS                                                                                                          ###
#################################################################################################################################
ROBOTS_GRASPS_LABELS = {
    'allegro_right': ALLEGRO_GRASPS_LABELS,
    'shadowhand': SHADOWHAND_GRASPS_LABELS,
}

IDX_2_GRASP_TYPE = {
    'allegro_right': {idx: grasp for idx, grasp in enumerate(ALLEGRO_GRASPS_LABELS.keys())},
    'shadowhand': {idx: grasp for idx, grasp in enumerate(SHADOWHAND_GRASPS_LABELS.keys())},
}

GRASP_TYPE_2_IDX = {
    'allegro_right': {grasp: idx for idx, grasp in enumerate(ALLEGRO_GRASPS_LABELS.keys())},
    'shadowhand': {grasp: idx for idx, grasp in enumerate(SHADOWHAND_GRASPS_LABELS.keys())},
}

ROBOTS_PALM_LABELS = {
    'allegro_right': [1, 2, 3, 4, 5, 6, 7],
    'shadowhand': [1, 2, 3, 4, 5, 6, 7],
}

ROBOTS_LINK_NAMES = {
    'allegro_right': ALLEGRO_LINK_NAMES,
    'shadowhand': SHADOWHAND_LINK_NAMES,
}



#################################################################################################################################
### TAXONOMY TRANSFER                                                                                                         ###
#################################################################################################################################

TAX_2_FEIX = {
    'm1': ['23_Adduction_Grip'],
    'm2': ['16_Lateral'],
    'm3': ['9_Palmar_Pinch', '24_Tip_Pinch'],
    'm4': ['8_Prismatic_2_Finger', '14_Tripod'],
    'm5': ['7_Prismatic_3_Finger', '27_Quadpod'],
    'm6': ['6_Prismatic_4_Finger', '12_Precision_Disk', '13_Precision_Sphere'],
    'm7': ['19_Distal_Type'],
    'm8': ['25_Lateral_Tripod'],
    'm9': ['20_Writing_Tripod'],
    'm10': ['21_Tripod_Variation'],
    'm11': ['29_Stick', '32_Ventral'],
    'm12': ['33_Inferior_Pincer'],
    'm13': ['31_Ring'],
    'm14': ['15_Fixed_Hook'],
    'm15': ['22_Parallel_Extension'],
    'm16': ['5_Light_Tool'],
    'm17': ['18_Extension_Type'],
    'm18': ['28_Sphere_3_Finger'],
    'm19': ['26_Sphere_4_Finger'],
    'm20': ['30_Palmar'],
    'm21': ['1_Large_Diameter', '2_Small_Diameter', '3_Medium_Wrap', '4_Adducted_Thumb', '10_Power_Disk', '11_Power_Sphere', '17_Index_Finger_Extension'],
}

FEIX_2_TAX = {
    '1_Large_Diameter': 'm21',
    '2_Small_Diameter': 'm21',
    '3_Medium_Wrap': 'm21',
    '4_Adducted_Thumb': 'm21',
    '5_Light_Tool': 'm16',
    '6_Prismatic_4_Finger': 'm6',
    '7_Prismatic_3_Finger': 'm5',
    '8_Prismatic_2_Finger': 'm4',
    '9_Palmar_Pinch': 'm3',
    '10_Power_Disk': 'm21',
    '11_Power_Sphere': 'm21',
    '12_Precision_Disk': 'm6',
    '13_Precision_Sphere': 'm6',
    '14_Tripod': 'm4',
    '15_Fixed_Hook': 'm14',
    '16_Lateral': 'm2',
    '17_Index_Finger_Extension': 'm21',
    '18_Extension_Type': 'm17',
    '19_Distal_Type': 'm7',
    '20_Writing_Tripod': 'm9',
    '21_Tripod_Variation': 'm10',
    '22_Parallel_Extension': 'm15',
    '23_Adduction_Grip': 'm1',
    '24_Tip_Pinch': 'm3',
    '25_Lateral_Tripod': 'm8',
    '26_Sphere_4_Finger': 'm19',
    '27_Quadpod': 'm5',
    '28_Sphere_3_Finger': 'm18',
    '29_Stick': 'm11',
    '30_Palmar': 'm20',
    '31_Ring': 'm13',
    '32_Ventral': 'm11',
    '33_Inferior_Pincer': 'm12'
}