from __future__ import annotations

import pandas as pd

PRODUCTS = [
    {"product_id":"AV1001","name":"Everyday Cloud Sports Bra","category":"Sports Bra","price":1499,"color":"Black","material":"Nylon-Elastane","support":"Low Impact","padding":"Removable","description":"Soft seamless sports bra designed for yoga, walking and everyday wear."},
    {"product_id":"AV1002","name":"Sculpt Medium Support Bra","category":"Sports Bra","price":1899,"color":"Black","material":"Polyester-Elastane","support":"Medium Impact","padding":"Fixed","description":"Supportive active bra with wide straps for gym training and studio workouts."},
    {"product_id":"AV1003","name":"AirFlex Yoga Bra","category":"Sports Bra","price":1699,"color":"Mauve","material":"Nylon-Elastane","support":"Low Impact","padding":"Removable","description":"Lightweight breathable yoga bra with soft-touch fabric and flexible straps."},
    {"product_id":"AV2001","name":"Contour High-Rise Leggings","category":"Leggings","price":2199,"color":"Black","material":"Nylon-Elastane","support":"Compression","padding":"N/A","description":"High-rise leggings with smooth compression and moisture-wicking fabric."},
    {"product_id":"AV2002","name":"Studio Soft Leggings","category":"Leggings","price":1799,"color":"Navy","material":"Polyester-Elastane","support":"Light Compression","padding":"N/A","description":"Soft stretch leggings designed for yoga, travel and lounge wear."},
    {"product_id":"AV3001","name":"CloudSoft Lounge Tee","category":"T-Shirt","price":999,"color":"White","material":"Cotton-Modal","support":"N/A","padding":"N/A","description":"Relaxed everyday tee made with a soft cotton-modal blend."},
    {"product_id":"AV4001","name":"Velocity Training Shoes","category":"Shoes","price":3299,"color":"Black","material":"Mesh-Synthetic","support":"Cushioned","padding":"N/A","description":"Breathable cross-training shoes with cushioned midsole and flexible grip."},
    {"product_id":"AV5001","name":"Recovery Slides","category":"Shoes","price":1399,"color":"Pink","material":"EVA","support":"Cushioned","padding":"N/A","description":"Lightweight cushioned slides for post-workout recovery and casual wear."},
]

REVIEW_TEMPLATES = {
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
    ],
    "AV2002": [
        (5,"Extremely soft and comfortable for yoga and travel."),
        (4,"Light compression and true to size."),
        (3,"Great comfort but not enough compression for intense training."),
        (5,"Feels like a second skin."),
    ],
    "AV3001": [
        (5,"Very soft fabric and relaxed fit."),
        (4,"Comfortable everyday tee and washes well."),
        (3,"Slightly oversized for me."),
    ],
    "AV4001": [
        (5,"Comfortable cushioning and good grip for gym workouts."),
        (4,"Breathable and supportive for cross training."),
        (3,"Runs slightly narrow in the toe box."),
        (5,"Lightweight and stable for strength training."),
    ],
    "AV5001": [
        (5,"Very cushioned and comfortable after workouts."),
        (4,"Lightweight and easy to wear."),
        (3,"A little wide for narrow feet."),
    ],
}

ASPECT_KEYWORDS = {
    "comfort": ["comfort", "soft", "cushion", "second skin", "hours"],
    "fit": ["fit", "size", "small", "tight", "narrow", "wide", "oversized"],
    "support": ["support", "secure", "compression", "stays in place", "stable"],
    "material": ["fabric", "breathable", "material", "mesh"],
    "activity": ["yoga", "gym", "training", "running", "workout", "stretching"],
    "padding": ["padding"],
    "durability": ["durable", "washes well"],
}


def load_products() -> pd.DataFrame:
    return pd.DataFrame(PRODUCTS)


def load_reviews() -> pd.DataFrame:
    rows = []
    idx = 1
    for product_id, templates in REVIEW_TEMPLATES.items():
        for cycle in range(4):
            for rating, text in templates:
                rows.append({
                    "review_id": f"R{idx:04d}",
                    "product_id": product_id,
                    "rating": rating,
                    "review_text": text,
                })
                idx += 1
    return pd.DataFrame(rows)
