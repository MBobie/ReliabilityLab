from reliabilitylab.data import load_banking77


dataset = load_banking77()

print(dataset)

print("\nFirst training sample:")
print(dataset["train"][0])

print("\nTraining samples:")
print(len(dataset["train"]))

print("\nTest samples:")
print(len(dataset["test"]))