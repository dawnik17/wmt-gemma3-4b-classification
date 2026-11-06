# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

INSTRUCTION = (
    "You are a product taxonomy classifier for an online retail catalog. Look at the product photo and "
    "read the listing that follows it, then assign the product to the single most specific category of "
    "the catalog taxonomy. Consider the packaging, the brand, the product form, the ingredients or "
    "materials and the intended use. Answer with the category name only."
)

PRODUCTS = {
    "cigarette_box": [
        "tobacco carton",
        "cigarette pack",
        "menthol cigarettes",
        "filter cigarettes",
    ],
    "drink_bottle": [
        "sparkling water",
        "iced tea",
        "sports drink",
        "fruit juice",
        "cold brew coffee",
    ],
    "drink_can": [
        "energy drink",
        "cola",
        "lemon soda",
        "ginger ale",
        "sparkling lemonade",
    ],
    "food_bottle": [
        "soy sauce",
        "salad dressing",
        "hot sauce",
        "olive oil",
        "maple syrup",
        "ketchup",
    ],
    "food_box": ["breakfast cereal", "crackers", "granola bars", "pasta", "cookies"],
    "food_package": [
        "potato chips",
        "trail mix",
        "gummy candy",
        "instant noodles",
        "jerky",
        "dried fruit",
    ],
}
BRANDS = [
    "Great Value",
    "Marketside",
    "Sam's Choice",
    "Northfield",
    "Blue Harbor",
    "Sunny Acres",
    "Golden Prairie",
    "Riverbend",
    "Maple Crest",
    "Evergreen Farms",
    "Coastal Kitchen",
    "Red Barn",
    "Silver Spoon",
    "Pine Hollow",
]
ADJECTIVES = [
    "classic",
    "original",
    "family size",
    "low sodium",
    "organic",
    "zero sugar",
    "extra crispy",
    "bold",
    "lightly salted",
    "reduced fat",
    "gluten free",
    "non-GMO",
    "fresh",
    "premium",
    "value pack",
]
MATERIALS = [
    "recyclable aluminum",
    "PET plastic",
    "glass",
    "paperboard",
    "a resealable foil pouch",
    "cardboard",
]
FLAVORS = [
    "sea salt",
    "barbecue",
    "cherry",
    "vanilla",
    "peach",
    "original",
    "spicy chili",
    "honey",
    "lime",
    "mint",
]
NAMES = [
    "Alex",
    "Jordan",
    "Sam",
    "Taylor",
    "Morgan",
    "Casey",
    "Riley",
    "Jamie",
    "Avery",
    "Quinn",
    "Drew",
    "Parker",
]
SENTENCES = [
    "This {adj} {product} from {brand} comes in {material} and is easy to store in a pantry or a fridge.",
    "Each {size} package of {product} is sealed at the factory to keep the {flavor} flavor fresh.",
    "The {product} is made with carefully selected ingredients and contains no artificial colors.",
    "Shoppers choose {brand} {product} for everyday meals, lunch boxes, road trips and parties.",
    "The label lists the nutrition facts, allergens and a best-before date printed near the {part}.",
    "Our {product} is produced in a facility that also processes milk, soy, wheat and tree nuts.",
    "For best results, store the {product} in a cool, dry place away from direct sunlight.",
    "The {material} packaging is designed to be stackable on store shelves and in delivery totes.",
    "Every case contains {count} units of {size} {product}, shrink wrapped for safe shipping.",
    "{brand} has been making {product} for over {years} years with the same {flavor} recipe.",
    "The {part} of the package shows the brand logo, the product name and a photo of the {product}.",
    "This item qualifies for pickup and delivery, and it may be substituted with a similar {product}.",
    "Once opened, keep the {product} tightly closed and consume it within {days} days.",
    "The {flavor} variety is one of the most popular choices in the {brand} line of {product}.",
    "Customers with dietary restrictions should read the ingredient list on the {part} before buying.",
    "The {product} pairs well with sandwiches, salads, snacks and other items from the same aisle.",
]
REVIEWS = [
    "I buy this {product} every week. The {flavor} taste is exactly what I expected and the price is fair.",
    "The package arrived with a dented {part}, but the {product} inside was fine and tasted great.",
    "Not my favorite. The {product} is a little too {taste} for me, but my kids love it.",
    "Great value for the {size} size. I keep a few of these in the pantry at all times.",
    "The {material} packaging is sturdy and the {product} stays fresh for a long time after opening.",
    "I compared it with other brands and {brand} is still the best {product} for the money.",
    "Delivery was quick and the {product} was cold when it arrived. Will order again.",
    "The new {part} design is nice, but I wish the {product} came in a bigger size.",
]
TASTES = ["sweet", "salty", "sour", "spicy", "bland", "rich", "strong", "mild"]
PARTS = ["front", "back", "side panel", "cap", "lid", "bottom", "seal", "label"]
SIZES = [8, 12, 16, 20, 24, 32, 48, 64]
COUNTS = [6, 12, 18, 24]


def listing(rng, category):
    """An endless product listing for one request; the caller cuts it to its token budget."""
    product = rng.choice(PRODUCTS[category])
    brand, adj, flavor = rng.choice(BRANDS), rng.choice(ADJECTIVES), rng.choice(FLAVORS)
    size = f"{rng.choice(SIZES)} oz"

    def fill(templates):
        return rng.choice(templates).format(
            product=product,
            brand=brand,
            adj=rng.choice(ADJECTIVES),
            size=size,
            flavor=flavor,
            material=rng.choice(MATERIALS),
            part=rng.choice(PARTS),
            taste=rng.choice(TASTES),
            count=rng.choice(COUNTS),
            years=rng.randint(5, 90),
            days=rng.randint(3, 30),
        )

    yield (
        f"Product listing\nTitle: {brand} {adj.title()} {flavor.title()} {product.title()}, {size}\n"
        f"Brand: {brand}\nItem ID: WM-{rng.randrange(10**8, 10**9)}\nPrice: ${rng.randint(1, 30)}.{rng.randint(0, 99):02d}\n"
        "Description:"
    )

    for _ in range(rng.randint(6, 9)):
        yield " " + fill(SENTENCES)

    yield "\nKey features:"
    for _ in range(5):
        yield "\n- " + fill(SENTENCES)

    yield (
        f"\nSpecifications:\n- Net content: {size}\n- Packaging: {rng.choice(MATERIALS)}\n- Flavor: {flavor}\n"
        f"- Units per case: {rng.choice(COUNTS)}\nCustomer reviews:"
    )
    while True:
        yield f"\n{rng.choice(NAMES)} ({rng.randint(1, 5)} stars): {fill(REVIEWS)} {fill(SENTENCES)}"
