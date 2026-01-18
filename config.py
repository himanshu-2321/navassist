# config.py

# ============ MODEL CONFIGURATION ============
MODEL = {
    'name': 'YOLOv8n',
    'weights': 'best.pt',
    'conf': 0.4,  # Increased confidence to reduce false alarms
}

# ============ RISK GROUPS (Source: 33-34) ============
# Defines how the system reacts to each group
RISK_CONFIG = {
    'G1': {'severity': 'VERY_HIGH', 'action': 'STOP'},    # Critical (Fire, Weapons)
    'G2': {'severity': 'LOW',       'action': 'INFO'},    # Navigation (Path, Stairs)
    'G3': {'severity': 'HIGH',      'action': 'REROUTE'}, # Construction (Holes, Cones)
    'G4': {'severity': 'MEDIUM',    'action': 'WARN'},    # Indoor (Chairs, Tables)
    'G6': {'severity': 'VERY_HIGH', 'action': 'STOP'},    # Traffic (Cars, Buses)
    'G7': {'severity': 'MED_HIGH',  'action': 'TRACK'},   # Living (People, Dogs)
    'G8': {'severity': 'LOW',       'action': 'INFO'}     # Static (Trees, Walls)
}

# ============ CLASS TO GROUP MAPPING (Source: 41-56) ============
# Maps YOLO classes to your Safety Groups
CLASS_TO_GROUP = {
    # G1: Critical
    'fire': 'G1', 'gun': 'G1', 'knife': 'G1', 'smoke': 'G1',
    
    # G2: Navigation
    'stairs': 'G2', 'curb': 'G2', 'door': 'G2',
    
    # G3: Construction
    'cone': 'G3', 'manhole': 'G3', 'excavation': 'G3',
    
    # G4: Indoor
    'chair': 'G4', 'table': 'G4', 'bottle': 'G4', 'laptop': 'G4',
    
    # G6: Traffic
    'car': 'G6', 'bus': 'G6', 'truck': 'G6', 'bicycle': 'G6', 'traffic light': 'G6',
    
    # G7: Living
    'person': 'G7', 'dog': 'G7', 'cat': 'G7',
    
    # G8: Static
    'tree': 'G8', 'pole': 'G8', 'fence': 'G8'
}

# ============ OBJECT HEIGHTS (Meters) (Source: 4-32) ============
# Used to calculate distance
OBJECT_HEIGHTS = {
    'person': 1.7, 'car': 1.5, 'bus': 3.0, 'truck': 3.0,
    'bike': 1.2, 'bicycle': 1.1, 'dog': 0.6, 'cat': 0.35,
    'chair': 0.9, 'table': 0.75, 'bottle': 0.25,
    'fire': 0.5, 'cone': 0.7, 'pole': 3.0, 'tree': 5.0,
    'default': 1.0
}