from src.data import load_products, load_reviews
from src.retrieval import extract_constraints
from src.intelligence import summarize_reviews
from src.graph import ask_victoria, route_intent


def test_data_loads():
    assert len(load_products()) >= 8
    assert len(load_reviews()) >= 50


def test_constraints():
    c = extract_constraints("Find me a black sports bra under ₹2000 for yoga")
    assert c["color"] == "black"
    assert c["category"] == "Sports Bra"
    assert c["max_price"] == 2000


def test_review_summary():
    products = load_products()
    reviews = load_reviews()
    pid = products.iloc[0]["product_id"]
    s = summarize_reviews(reviews[reviews["product_id"] == pid])
    assert s["review_count"] > 0
    assert s["average_rating"] > 0


def test_router():
    assert route_intent("Find me a bra under ₹2000") == "recommendation"
    assert route_intent("What do customers say about fit?") == "review"
    assert route_intent("Compare these products") == "comparison"


def test_end_to_end_recommendation():
    result = ask_victoria("Find me a black sports bra under ₹2000 for yoga")
    assert result["final_answer"]
    assert result["grounded"] is True
    assert result["intent"] == "recommendation"
