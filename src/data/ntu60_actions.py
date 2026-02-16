"""
NTU RGB+D 60 Action Class Mapping

Provides utilities to convert between action labels (0-59) and action names.
Maps to ntu60_actions.txt for reference.
"""

# NTU RGB+D 60 has 60 action classes (0-indexed: 0-59)
NTU60_ACTIONS = {
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
}


def get_action_name(label: int) -> str:
    """
    Convert action label to action name.
    
    Args:
        label: Action label (0-indexed, 0-59)
        
    Returns:
        Action name string
        
    Example:
        >>> get_action_name(9)
        'clapping'
    """
    if label not in NTU60_ACTIONS:
        return f"unknown (label {label})"
    return NTU60_ACTIONS[label]


def get_action_label(name: str) -> int:
    """
    Convert action name to label.
    
    Args:
        name: Action name (case-insensitive, partial match allowed)
        
    Returns:
        Action label (0-indexed)
        
    Raises:
        ValueError: If action name not found
    """
    name_lower = name.lower().strip()
    
    # Exact match
    for label, action_name in NTU60_ACTIONS.items():
        if action_name.lower() == name_lower:
            return label
    
    # Partial match
    for label, action_name in NTU60_ACTIONS.items():
        if name_lower in action_name.lower():
            return label
    
    raise ValueError(f"Action '{name}' not found in NTU60")


def get_num_classes() -> int:
    """Get number of action classes (60)."""
    return 60


def validate_label(label: int) -> bool:
    """Check if label is valid (0-59)."""
    return 0 <= label < 60
