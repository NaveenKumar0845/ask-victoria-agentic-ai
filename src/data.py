from __future__ import annotations

import pandas as pd

# Public-safe synthetic catalogue. The first eight products are stable demo anchors used
# by the UI, tests and interview walkthroughs; the rest are generated deterministically
# to create a more realistic retrieval corpus without proprietary retail data.
BASE_PRODUCTS = [
    {"product_id":"AV1001","name":"Everyday Cloud Sports Bra","category":"Sports Bra","price":1499,"color":"Black","material":"Nylon-Elastane","support":"Low Impact","padding":"Removable","description":"Soft seamless sports bra designed for yoga, walking and everyday wear."},
    {"product_id":"AV1002","name":"Sculpt Medium Support Bra","category":"Sports Bra","price":1899,"color":"Black","material":"Polyester-Elastane","support":"Medium Impact","padding":"Fixed","description":"Supportive active bra with wide straps for gym training and studio workouts."},
    {"product_id":"AV1003","name":"AirFlex Yoga Bra","category":"Sports Bra","price":1699,"color":"Mauve","material":"Nylon-Elastane","support":"Low Impact","padding":"Removable","description":"Lightweight breathable yoga bra with soft-touch fabric and flexible straps."},
    {"product_id":"AV2001","name":"Contour High-Rise Leggings","category":"Leggings","price":2199,"color":"Black","material":"Nylon-Elastane","support":"Compression","padding":"N/A","description":"High-rise leggings with smooth compression and moisture-wicking fabric."},
    {"product_id":"AV2002","name":"Studio Soft Leggings","category":"Leggings","price":1799,"color":"Navy","material":"Polyester-Elastane","support":"Light Compression","padding":"N/A","description":"Soft stretch leggings designed for yoga, travel and lounge wear."},
    {"product_id":"AV3001","name":"CloudSoft Lounge Tee","category":"T-Shirt","price":999,"color":"White","material":"Cotton-Modal","support":"N/A","padding":"N/A","description":"Relaxed everyday tee made with a soft cotton-modal blend."},
    {"product_id":"AV4001","name":"Velocity Training Shoes","category":"Shoes","price":3299,"color":"Black","material":"Mesh-Synthetic","support":"Cushioned","padding":"N/A","description":"Breathable cross-training shoes with cushioned midsole and flexible grip."},
    {"product_id":"AV5001","name":"Recovery Slides","category":"Shoes","price":1399,"color":"Pink","material":"EVA","support":"Cushioned","padding":"N/A","description":"Lightweight cushioned slides for post-workout recovery and casual wear."},
]

CATEGORY_SPECS = {
    "Sports Bra": {
        "names": ["MotionFlex Bra", "Balance Studio Bra", "Core Support Bra", "Breathe Easy Bra"],
        "colors": ["Navy", "White", "Pink", "Black"],
        "materials": ["Nylon-Elastane", "Polyester-Elastane", "Recycled Nylon", "Seamless Knit"],
        "supports": ["Low Impact", "Low Impact", "Medium Impact", "Medium Impact"],
        "paddings": ["Removable", "Removable", "Fixed", "Removable"],
        "prices": [1599, 1799, 2099, 1999],
        "descriptions": [
            "Flexible studio bra for yoga, Pilates and mobility sessions.",
            "Soft everyday sports bra with breathable fabric for studio workouts.",
            "Secure medium-support bra for strength training and gym sessions.",
            "Breathable seamless bra balancing comfort and moderate support.",
        ],
    },
    "Leggings": {
        "names": ["Flow High-Rise Leggings", "PowerForm Training Tights", "Everyday Pocket Leggings", "AirKnit Studio Leggings"],
        "colors": ["Mauve", "Black", "Navy", "Black"],
        "materials": ["Nylon-Elastane", "Recycled Polyester", "Polyester-Elastane", "Seamless Knit"],
        "supports": ["Light Compression", "Compression", "Light Compression", "Light Compression"],
        "paddings": ["N/A"] * 4,
        "prices": [1899, 2399, 1999, 2099],
        "descriptions": [
            "Soft high-rise leggings for yoga, stretching and low-impact movement.",
            "Supportive training tights with firm compression for gym workouts.",
            "Versatile leggings with side pockets for walking, travel and everyday wear.",
            "Breathable seamless leggings for studio sessions and relaxed movement.",
        ],
    },
    "T-Shirt": {
        "names": ["Everyday Modal Tee", "Active Mesh Tee", "Relaxed Cotton Tee", "Studio Crop Tee"],
        "colors": ["Black", "White", "Pink", "Navy"],
        "materials": ["Cotton-Modal", "Recycled Polyester", "Cotton", "Cotton-Modal"],
        "supports": ["N/A"] * 4,
        "paddings": ["N/A"] * 4,
        "prices": [1099, 1299, 899, 1199],
        "descriptions": [
            "Soft modal-blend tee for everyday layering and lounge wear.",
            "Breathable performance tee for gym sessions and warm-weather training.",
            "Relaxed cotton tee with an easy everyday fit.",
            "Soft cropped tee for studio classes, casual styling and layering.",
        ],
    },
    "Shorts": {
        "names": ["Studio Bike Shorts", "Flex Training Shorts", "Everyday Lounge Shorts", "RunLite Shorts"],
        "colors": ["Black", "Navy", "Mauve", "Black"],
        "materials": ["Nylon-Elastane", "Recycled Polyester", "Cotton-Modal", "Polyester-Elastane"],
        "supports": ["Compression", "Light Compression", "N/A", "Light Compression"],
        "paddings": ["N/A"] * 4,
        "prices": [1499, 1599, 1199, 1699],
        "descriptions": [
            "High-rise bike shorts with smooth compression for studio and cycling workouts.",
            "Flexible training shorts for strength sessions and functional movement.",
            "Soft lounge shorts for travel, recovery and relaxed everyday wear.",
            "Lightweight running shorts with breathable stretch fabric and easy movement.",
        ],
    },
    "Jacket": {
        "names": ["CloudZip Jacket", "Studio Wrap Jacket", "RunShell Jacket", "Everyday Knit Jacket"],
        "colors": ["Black", "Mauve", "Navy", "White"],
        "materials": ["Polyester-Blend", "Nylon-Blend", "Recycled Polyester", "Cotton-Blend"],
        "supports": ["N/A"] * 4,
        "paddings": ["N/A"] * 4,
        "prices": [2799, 2499, 3299, 2599],
        "descriptions": [
            "Soft full-zip layer for warm-ups, travel and everyday activewear styling.",
            "Light studio wrap designed for layering before and after low-impact sessions.",
            "Lightweight outer shell for outdoor training and windy conditions.",
            "Comfortable knit jacket for casual layering and recovery days.",
        ],
    },
    "Sleepwear": {
        "names": ["DreamSoft Pajama Set", "Modal Sleep Tee", "Cloud Lounge Joggers", "Relaxed Sleep Shorts"],
        "colors": ["Pink", "Navy", "Black", "Mauve"],
        "materials": ["Cotton-Modal", "Modal", "Cotton-Modal", "Cotton"],
        "supports": ["N/A"] * 4,
        "paddings": ["N/A"] * 4,
        "prices": [2299, 1199, 1799, 999],
        "descriptions": [
            "Soft pajama set designed for breathable overnight comfort.",
            "Lightweight modal sleep tee with a relaxed drape.",
            "Soft lounge joggers for home, travel and recovery days.",
            "Relaxed cotton sleep shorts with an easy elastic waist.",
        ],
    },
    "Shoes": {
        "names": ["Studio Trainer", "FlexWalk Sneakers", "LiftStable Training Shoes", "CloudRun Trainers"],
        "colors": ["White", "Navy", "Black", "Pink"],
        "materials": ["Mesh-Synthetic", "Knit-Synthetic", "Mesh-Synthetic", "Engineered Mesh"],
        "supports": ["Cushioned", "Cushioned", "Stable", "Cushioned"],
        "paddings": ["N/A"] * 4,
        "prices": [2999, 2899, 3499, 3599],
        "descriptions": [
            "Versatile studio trainer with flexible cushioning for classes and gym sessions.",
            "Comfort-focused walking sneaker with breathable knit upper.",
            "Stable training shoe designed for strength work and controlled gym movement.",
            "Lightweight trainer with responsive cushioning for short runs and cardio sessions.",
        ],
    },
    "Accessories": {
        "names": ["Studio Grip Socks", "Everyday Tote", "Training Headband Set", "Active Bottle Sling"],
        "colors": ["Black", "Pink", "Navy", "Black"],
        "materials": ["Cotton-Blend", "Recycled Polyester", "Nylon-Blend", "Recycled Polyester"],
        "supports": ["Grip", "N/A", "Stretch", "N/A"],
        "paddings": ["N/A"] * 4,
        "prices": [699, 1299, 599, 899],
        "descriptions": [
            "Soft studio socks with grip zones for Pilates and low-impact classes.",
            "Lightweight everyday tote for gym essentials and casual use.",
            "Stretch headband set designed to stay comfortable during workouts.",
            "Compact bottle sling for walks, commuting and light training days.",
        ],
    },
}


def _generated_products() -> list[dict]:
    products: list[dict] = []
    category_prefix = {
        "Sports Bra": "AV11",
        "Leggings": "AV21",
        "T-Shirt": "AV31",
        "Shorts": "AV61",
        "Jacket": "AV71",
        "Sleepwear": "AV81",
        "Shoes": "AV41",
        "Accessories": "AV91",
    }
    for category, spec in CATEGORY_SPECS.items():
        for idx, name in enumerate(spec["names"], start=1):
            products.append(
                {
                    "product_id": f"{category_prefix[category]}{idx:02d}",
                    "name": name,
                    "category": category,
                    "price": spec["prices"][idx - 1],
                    "color": spec["colors"][idx - 1],
                    "material": spec["materials"][idx - 1],
                    "support": spec["supports"][idx - 1],
                    "padding": spec["paddings"][idx - 1],
                    "description": spec["descriptions"][idx - 1],
                }
            )
    return products


PRODUCTS = BASE_PRODUCTS + _generated_products()

ASPECT_KEYWORDS = {
    "comfort": ["comfort", "soft", "cushion", "second skin", "hours", "relaxed"],
    "fit": ["fit", "size", "small", "tight", "narrow", "wide", "oversized", "true to size"],
    "support": ["support", "secure", "compression", "stays in place", "stable", "firm"],
    "material": ["fabric", "breathable", "material", "mesh", "modal", "cotton", "knit"],
    "activity": ["yoga", "gym", "training", "running", "workout", "stretching", "walking", "studio"],
    "padding": ["padding", "padded"],
    "durability": ["durable", "washes well", "holds up", "quality"],
    "style": ["flattering", "style", "color", "looks", "cute"],
}

BASE_REVIEW_TEMPLATES = {
    "AV1001": [
        (5,"Very soft and comfortable for yoga. I can wear it for hours."),
        (4,"Comfortable and flattering, but the band runs a little small."),
        (5,"Love the soft fabric and removable padding."),
        (3,"Good for low impact workouts, but I would size up."),
        (5,"Great everyday bra and very breathable."),
    ],
    "AV1002": [
        (5,"Excellent support for gym workouts and the wide straps feel secure."),
        (4,"Supportive and true to size, though the fixed padding is not my favorite."),
        (4,"Good medium support and stays in place during training."),
        (3,"A little firm around the band but supportive."),
        (5,"Great fit and strong support without digging into shoulders."),
    ],
    "AV1003": [
        (5,"Super lightweight and perfect for yoga."),
        (4,"Soft and breathable with comfortable straps."),
        (4,"Pretty color and flexible fit, support is definitely light."),
        (3,"Comfortable but not enough support for running."),
        (5,"My favorite for stretching and low impact classes."),
    ],
    "AV2001": [
        (5,"Great compression and stays up during workouts."),
        (4,"Smooth fabric and flattering fit."),
        (3,"Waistband feels tight after a few hours."),
        (5,"Excellent for training and very durable."),
        (4,"Supportive high-rise fit and breathable enough for the gym."),
    ],
    "AV2002": [
        (5,"Extremely soft and comfortable for yoga and travel."),
        (4,"Light compression and true to size."),
        (3,"Great comfort but not enough compression for intense training."),
        (5,"Feels like a second skin."),
        (4,"Soft fabric works well for stretching and everyday wear."),
    ],
    "AV3001": [
        (5,"Very soft fabric and relaxed fit."),
        (4,"Comfortable everyday tee and washes well."),
        (3,"Slightly oversized for me."),
        (5,"The cotton modal fabric feels smooth and breathable."),
        (4,"Easy lounge tee with a flattering relaxed shape."),
    ],
    "AV4001": [
        (5,"Comfortable cushioning and good grip for gym workouts."),
        (4,"Breathable and supportive for cross training."),
        (3,"Runs slightly narrow in the toe box."),
        (5,"Lightweight and stable for strength training."),
        (4,"Good gym shoe with comfortable cushioning for longer sessions."),
    ],
    "AV5001": [
        (5,"Very cushioned and comfortable after workouts."),
        (4,"Lightweight and easy to wear."),
        (3,"A little wide for narrow feet."),
        (5,"Soft underfoot feel is great for recovery days."),
        (4,"Easy casual slide and the cushioning feels supportive."),
    ],
}

POSITIVE_REVIEW_PATTERNS = [
    "The {material} feels soft and comfortable, and the fit works well for {activity}.",
    "Really like the {support} feel. It stays comfortable through my {activity} sessions.",
    "Breathable material and a flattering fit. I would wear this for {activity} again.",
    "Good quality for the price and the fabric holds up well after regular use.",
    "Comfort is the biggest strength for me; the design feels easy to wear for hours.",
]
MIXED_REVIEW_PATTERNS = [
    "Comfortable overall, although the fit feels a little snug compared with what I expected.",
    "I like the material and style, but sizing may be worth checking carefully.",
    "Good for {activity}, though I wanted slightly more support for intense movement.",
    "Soft fabric and nice quality, but the cut felt a bit wide on me.",
    "Looks good and feels comfortable, though the fit may depend on body shape.",
]

CATEGORY_ACTIVITY = {
    "Sports Bra": "yoga and gym",
    "Leggings": "studio and training",
    "T-Shirt": "everyday wear and light workouts",
    "Shorts": "training and warm-weather workouts",
    "Jacket": "warm-ups and everyday layering",
    "Sleepwear": "sleep and lounge",
    "Shoes": "walking and gym training",
    "Accessories": "studio and everyday use",
}


def _reviews_for_product(product: dict, reviews_per_product: int = 20) -> list[tuple[int, str]]:
    if product["product_id"] in BASE_REVIEW_TEMPLATES:
        base = BASE_REVIEW_TEMPLATES[product["product_id"]]
    else:
        base = []
    activity = CATEGORY_ACTIVITY.get(product["category"], "everyday use")
    generated: list[tuple[int, str]] = list(base)
    patterns = POSITIVE_REVIEW_PATTERNS + MIXED_REVIEW_PATTERNS
    idx = 0
    while len(generated) < reviews_per_product:
        pattern = patterns[idx % len(patterns)]
        rating = [5, 4, 5, 4, 5, 4, 3, 4, 3, 4][idx % 10]
        text = pattern.format(
            material=product["material"].lower(),
            support=product["support"].lower(),
            activity=activity,
        )
        # Deterministic light variation prevents a corpus made of exact duplicates.
        if idx % 3 == 1:
            text += " The color also looks close to the product photos."
        elif idx % 3 == 2:
            text += " I would consider buying another color."
        generated.append((rating, text))
        idx += 1
    return generated[:reviews_per_product]


def load_products() -> pd.DataFrame:
    return pd.DataFrame(PRODUCTS)


def load_reviews() -> pd.DataFrame:
    rows = []
    review_idx = 1
    for product in PRODUCTS:
        for rating, text in _reviews_for_product(product, reviews_per_product=20):
            rows.append(
                {
                    "review_id": f"R{review_idx:05d}",
                    "product_id": product["product_id"],
                    "rating": rating,
                    "review_text": text,
                }
            )
            review_idx += 1
    return pd.DataFrame(rows)
