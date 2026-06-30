"""The phrasings we wrap each option in before embedding it.

We don't compare the image to a bare word like "leather" — we compare it to a few
natural sentences ("a photo of a shoe made of leather", ...) and average them. That
smooths out the quirks of any single wording and keeps generic labels from winning by
accident. {category} and {option} are filled in per call.
"""

# Per-attribute phrasings. Each attribute reads better with wording tuned to it.
TEMPLATES = {
    "color":    ["a photo of a {option} {category}", "a {option} {category}", "a {option}-colored {category}"],
    "material": ["a photo of a {category} made of {option}", "a {option} {category}",
                 "a close-up of {option} texture", "a {category} in {option}"],
    # one tight phrasing — looser ones drift to trendy labels (formal->luxury)
    "style":    ["a photo of a {option}-style {category}"],
    "gender":   ["a photo of a {category} for {option}", "a photo of a {option}'s {category}"],
    "pattern":  ["a photo of a {category} with a {option} pattern", "a {option} {category}",
                 "a photo of a {category} with {option} print"],
}

# Used for any attribute without its own entry above.
DEFAULT_TEMPLATES = ["a photo of {option} {category}", "a {option} {category}"]

# Phrasings for choosing the category itself (no {category} to fill in here).
CATEGORY_TEMPLATES = ["a photo of {option}", "a photo of a {option}", "a product photo of {option}"]
