SEQ_INT2STR = {
    "0": "B",
    "1": "S",
    "2": "R",
    "3": "O",
}

SEQ_STR2INT = {
    "B": 0,
    "S": 1,
    "R": 2,
    "O": 3,
}

TAG_INT2STR = SEQ_INT2STR
TAG_STR2INT = SEQ_STR2INT

# Scheduler-specific absorbing state for mask-based diffusion kernels.
MASK_STATE_ID = 4
