"""
NTU RGB+D 120 Action Class Mapping

Provides utilities to convert between action labels (0-119) and action names.
"""

# NTU RGB+D 120 has 120 action classes (0-indexed: 0-119)
NTU120_ACTIONS = {
    0: "drink water",
    1: "eat meal/snack",
    2: "brushing teeth",
    3: "brushing hair",
    4: "drop",
    5: "pickup",
    6: "throw",
    7: "sitting down",
    8: "standing up (from sitting position)",
    9: "clapping",
    10: "reading",
    11: "writing",
    12: "tear up paper",
    13: "wear jacket",
    14: "take off jacket",
    15: "wear a shoe",
    16: "take off a shoe",
    17: "wear on glasses",
    18: "take off glasses",
    19: "put on a hat/cap",
    20: "take off a hat/cap",
    21: "cheer up",
    22: "hand waving",
    23: "kicking something",
    24: "reach into pocket",
    25: "hopping (one foot jumping)",
    26: "jump up",
    27: "make a phone call/answer phone",
    28: "playing with phone/tablet",
    29: "typing on a keyboard",
    30: "pointing to something with finger",
    31: "taking a selfie",
    32: "check time (from watch)",
    33: "rub two hands together",
    34: "nod head/bow",
    35: "shake head",
    36: "wipe face",
    37: "salute",
    38: "put the palms together",
    39: "cross hands in front (say stop)",
    40: "sneeze/cough",
    41: "staggering",
    42: "falling",
    43: "touch head (headache)",
    44: "touch chest (stomachache/heart pain)",
    45: "touch back (backache)",
    46: "touch neck (neckache)",
    47: "nausea or vomiting condition",
    48: "use a fan (with hand or paper)/feeling warm",
    49: "punching/slapping other person",
    50: "kicking other person",
    51: "pushing other person",
    52: "pat on back of other person",
    53: "point finger at the other person",
    54: "hugging other person",
    55: "giving something to other person",
    56: "touch other person's pocket",
    57: "handshaking",
    58: "walking towards each other",
    59: "walking apart from each other",
    60: "put on headphone",
    61: "take off headphone",
    62: "shoot at the basket",
    63: "bounce ball",
    64: "tennis bat swing",
    65: "juggling table tennis balls",
    66: "hush (quite)",
    67: "flick hair",
    68: "thumb up",
    69: "thumb down",
    70: "make ok sign",
    71: "make victory sign",
    72: "staple book",
    73: "counting money",
    74: "cutting nails",
    75: "cutting paper (using scissors)",
    76: "snapping fingers",
    77: "open bottle",
    78: "sniff (smell)",
    79: "squat down",
    80: "toss a coin",
    81: "fold paper",
    82: "ball up paper",
    83: "play magic cube",
    84: "apply cream on face",
    85: "apply cream on hand back",
    86: "put on bag",
    87: "take off bag",
    88: "put something into a bag",
    89: "take something out of a bag",
    90: "open a box",
    91: "move heavy objects",
    92: "shake fist",
    93: "throw up cap/hat",
    94: "hands up (both hands)",
    95: "cross arms",
    96: "arm circles",
    97: "arm swings",
    98: "running on the spot",
    99: "butt kicks (kick backward)",
    100: "cross toe touch",
    101: "side kick",
    102: "yawn",
    103: "stretch oneself",
    104: "blow nose",
    105: "hit other person with something",
    106: "wield knife towards other person",
    107: "knock over other person (hit with body)",
    108: "grab other person's stuff",
    109: "shoot at other person with a gun",
    110: "step on foot",
    111: "high-five",
    112: "cheers and drink",
    113: "carry something with other person",
    114: "take a photo of other person",
    115: "follow other person",
    116: "whisper in other person's ear",
    117: "exchange things with other person",
    118: "support somebody with hand",
    119: "finger-guessing game (playing rock-paper-scissors)",
}

# NTU RGB+D 60 subset (first 60 actions)
NTU60_ACTIONS = {k: v for k, v in NTU120_ACTIONS.items() if k < 60}


def get_action_name(label: int, dataset: str = 'ntu120') -> str:
    """
    Convert action label to action name.
    
    Args:
        label: Action label (0-indexed)
        dataset: 'ntu60' or 'ntu120'
        
    Returns:
        Action name string
        
    Example:
        >>> get_action_name(12)
        'tear up paper'
    """
    action_map = NTU60_ACTIONS if dataset == 'ntu60' else NTU120_ACTIONS
    
    if label not in action_map:
        return f"unknown (label {label})"
    
    return action_map[label]


def get_action_label(name: str, dataset: str = 'ntu120') -> int:
    """
    Convert action name to label.
    
    Args:
        name: Action name (case-insensitive, partial match allowed)
        dataset: 'ntu60' or 'ntu120'
        
    Returns:
        Action label (0-indexed)
        
    Raises:
        ValueError: If action name not found
        
    Example:
        >>> get_action_label('tear up paper')
        12
    """
    action_map = NTU60_ACTIONS if dataset == 'ntu60' else NTU120_ACTIONS
    
    name_lower = name.lower().strip()
    
    # Exact match
    for label, action_name in action_map.items():
        if action_name.lower() == name_lower:
            return label
    
    # Partial match
    for label, action_name in action_map.items():
        if name_lower in action_name.lower():
            return label
    
    raise ValueError(f"Action '{name}' not found in {dataset.upper()}")


def get_num_classes(dataset: str = 'ntu120') -> int:
    """
    Get number of action classes.
    
    Args:
        dataset: 'ntu60' or 'ntu120'
        
    Returns:
        Number of classes
    """
    return 60 if dataset == 'ntu60' else 120


def validate_label(label: int, dataset: str = 'ntu120') -> bool:
    """
    Check if label is valid for the dataset.
    
    Args:
        label: Action label to validate
        dataset: 'ntu60' or 'ntu120'
        
    Returns:
        True if valid, False otherwise
    """
    num_classes = get_num_classes(dataset)
    return 0 <= label < num_classes
