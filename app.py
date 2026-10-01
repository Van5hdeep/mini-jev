
import time
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

t0 = time.time()

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
)

classifier("warm up", candidate_labels=["a", "b"])

print(f"model ready in {time.time() - t0:.0f}s")


ROUTES = ["billing", "technical problem", "sales question", "thank you"]

def choice(text, options, template="this message is about {}."):
    r = classifier(
        text,
        candidate_labels=options,
        hypothesis_template=template
    )

    return {
        label: round(p, 3)
        for label, p in zip(r["labels"], r["scores"])
    }


def noun(text, yes, no, template="The customer is {}."):
    return choice(text, [yes, no], template)[yes]


URGENCY = [
    "it can wait a week",
    "it should be handled today",
    "it needs action right now"
]

def score(text, levels, template="for support, {}."):
    probs = choice(text, levels, template)

    value = sum(
        (levels.index(level) + 1) * p
        for level, p in probs.items()
    )

    return round(value, 2)


if __name__ == "__main__":

    print(choice(
        "my laptop started malfunctioning.",
        ROUTES
    ))

    print(choice(
        "I was charged twice this month and nobody is answering",
        ROUTES
    ))

    print(
        "angry:",
        noun(
            "fix my laptop now!",
            "angry or frustrated",
            "calm or happy"
        )
    )

    print(
        "angry:",
        noun(
            "you are a life saver, thank you",
            "angry or frustrated",
            "calm or happy"
        )
    )

    for m in [
        "our whole company is locked out of the dashboard since morning, we cant work.",
        "we are in need of immediate medical attention, my friend cannot breath!",
        "thank you, the fix worked just fine."
    ]:
        print(f"urgency {score(m, URGENCY)} / 3 <-{m}")
