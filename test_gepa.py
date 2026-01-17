"""Minimal GEPA optimization test with emotion classification.

Demonstrates GEPA optimizer with metric_with_feedback for per-predictor feedback.
Saves/loads optimized model to emotion_classifier_optimized.json.
"""
import dspy
from dspy.teleprompt import GEPA

if __name__ == '__main__':
    # Setup LM
    lm = dspy.LM(
        model="ollama/llama3.2-vision:11b",
        api_base="http://localhost:11434",
        temperature=0.0
    )
    dspy.configure(lm=lm)

    # Training data
    train = [
        dspy.Example(sentence="I love this!", emotion="happy").with_inputs("sentence"),
        dspy.Example(sentence="This is terrible.", emotion="sad").with_inputs("sentence"),
        dspy.Example(sentence="The weather is okay.", emotion="neutral").with_inputs("sentence"),
    ]

    val = [
        dspy.Example(sentence="I'm so excited!", emotion="happy").with_inputs("sentence"),
        dspy.Example(sentence="I hate Mondays.", emotion="sad").with_inputs("sentence"),
        dspy.Example(sentence="The sky is blue.", emotion="neutral").with_inputs("sentence"),
    ]

    # Module
    class EmotionClassifier(dspy.Module):
        def __init__(self):
            self.predict = dspy.ChainOfThought("sentence -> emotion")
        
        def forward(self, sentence):
            return self.predict(sentence=sentence)

    # Metric with feedback
    def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
        correct = example.emotion.lower() == pred.emotion.lower()
        score = 1.0 if correct else 0.0
        
        if pred_name is None:
            return score
        
        if correct:
            feedback = f"Correct! You classified '{example.sentence}' as '{pred.emotion}', which matches the gold label '{example.emotion}'."
        else:
            feedback = f"Incorrect. You classified '{example.sentence}' as '{pred.emotion}', but the correct emotion is '{example.emotion}'. Think about the emotional tone more carefully."
        
        return dspy.Prediction(score=score, feedback=feedback)

    # Optimize
    print("Starting GEPA optimization...")
    optimizer = GEPA(
        metric=metric_with_feedback,
        auto="light",
        num_threads=4,
        track_stats=True,
        reflection_lm=dspy.LM(
            model="ollama/llama3.2-vision:11b",
            api_base="http://localhost:11434",
            temperature=1.0
        )
    )

    optimized = optimizer.compile(
        EmotionClassifier(),
        trainset=train,
        valset=val
    )
    
    # Save optimized model
    optimized.save("emotion_classifier_optimized.json")
    print("\nSaved optimized model to emotion_classifier_optimized.json")

    # Load and test (comment out optimization above to test loading)
    # optimized = EmotionClassifier()
    # optimized.load("emotion_classifier_optimized.json")
    # print("\nLoaded optimized model from emotion_classifier_optimized.json")
    
    # Test
    print("\n=== Testing Optimized Model ===")
    test_cases = ["I love pizza", "I hate Mondays", "The sky is blue"]
    for test in test_cases:
        result = optimized(sentence=test)
        print(f"'{test}' -> {result.emotion}")

    # Show optimized prompt
    print("\n=== Optimized Prompt ===")
    for name, pred in optimized.named_predictors():
        print(f"\n{name}:")
        print(pred.signature.instructions)
