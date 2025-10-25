# Image type specs from your guidelines
IMAGE_TYPES = {
    "offer_tile":   {"label": "Offer Tile",   "size": (1120, 1120), "format": "jpg",  "allow_lifestyle": False},
    "offer_detail": {"label": "Offer Detail", "size": (1120, 1120), "format": "jpeg", "allow_lifestyle": True},
    "brand_hero":   {"label": "Brand Hero",   "size": (1200, 857),  "format": "jpg",  "allow_lifestyle": True},
    "brand_logo":   {"label": "Brand Logo",   "size": (400, 400),   "format": "png",  "circle_safe": True}
}

# Approved background palette
APPROVED_SWATCHES = [
    {"name":"Pink","hex":"#F7CAD0"},
    {"name":"Orange","hex":"#FBB040"},
    {"name":"Yellow","hex":"#FFF275"},
    {"name":"Yellow-Green","hex":"#C4E86B"},
    {"name":"Green","hex":"#7ED957"},
    {"name":"Blue-Green","hex":"#74D4C0"},
    {"name":"Blue","hex":"#72A1E5"},
    {"name":"Purple","hex":"#A785E4"},
    {"name":"Violet","hex":"#CFA9E9"},
    {"name":"Cool Gray","hex":"#C3C7C8"},
    {"name":"Warm Gray","hex":"#C6BBAE"},
    {"name":"Beige","hex":"#E7D5B0"},
]
APPROVED_HEX = {c["hex"].upper() for c in APPROVED_SWATCHES}
