extends Node

# Simple test script for coordinated shuffle functionality

func _ready() -> void:
	print("=== Testing Coordinated Shuffle ===")
	
	# Test 1: Basic coordinated shuffle
	print("\nTest 1: Basic 5-card deck shuffle")
	var shuffled_deck = StatMath.SamplingGen.coordinated_shuffle(5, StatMath.SamplingGen.SamplingMethod.SOBOL, 0)
	print("Shuffled deck: ", shuffled_deck)
	
	# Test 2: Multiple coordinated shuffles should be different
	print("\nTest 2: Multiple shuffles should be different")
	var shuffle1 = StatMath.SamplingGen.coordinated_shuffle(10, StatMath.SamplingGen.SamplingMethod.SOBOL, 0)
	var shuffle2 = StatMath.SamplingGen.coordinated_shuffle(10, StatMath.SamplingGen.SamplingMethod.SOBOL, 1)
	var shuffle3 = StatMath.SamplingGen.coordinated_shuffle(10, StatMath.SamplingGen.SamplingMethod.SOBOL, 2)
	print("Shuffle 1: ", shuffle1)
	print("Shuffle 2: ", shuffle2)
	print("Shuffle 3: ", shuffle3)
	
	# Test 3: Unified generate_samples interface
	print("\nTest 3: Unified generate_samples interface")
	var samples_1d = StatMath.SamplingGen.generate_samples(5, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var samples_2d = StatMath.SamplingGen.generate_samples(3, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var samples_5d = StatMath.SamplingGen.generate_samples(2, 5, StatMath.SamplingGen.SamplingMethod.SOBOL)
	print("1D samples: ", samples_1d)
	print("2D samples: ", samples_2d)
	print("5D samples: ", samples_5d)
	
	# Test 4: COORDINATED_FISHER_YATES selection strategy (smaller deck for performance)
	print("\nTest 4: COORDINATED_FISHER_YATES selection strategy")
	print("Testing with 10-card deck first...")
	var indices_small = StatMath.SamplingGen.sample_indices(
		10, 5, 
		StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES,
		StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	print("Coordinated indices (10-card deck, 5 cards): ", indices_small)
	
	# Test 4b: Try with a 20-card deck
	print("Testing with 20-card deck...")
	var indices_medium = StatMath.SamplingGen.sample_indices(
		20, 5, 
		StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES,
		StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	print("Coordinated indices (20-card deck, 5 cards): ", indices_medium)
	
	# Test 5: Batch coordinated shuffles
	print("\nTest 5: Batch coordinated shuffles")
	var batch_shuffles = StatMath.SamplingGen.coordinated_batch_shuffles(5, 3, StatMath.SamplingGen.SamplingMethod.SOBOL)
	for i in range(batch_shuffles.size()):
		print("Batch shuffle ", i, ": ", batch_shuffles[i])
	
	# Test 6: Demonstrate your royal flush use case
	print("\nTest 6: Royal flush simulation setup (proof of concept)")
	print("Generating 5 coordinated shuffles for royal flush analysis:")
	for trial in range(5):
		var deck_shuffle = StatMath.SamplingGen.coordinated_shuffle(52, StatMath.SamplingGen.SamplingMethod.SOBOL, trial)
		var hand = deck_shuffle.slice(0, 5)  # First 5 cards
		print("Trial ", trial, " hand: ", hand)
	
	print("\n=== All tests completed ===")
	get_tree().quit() 
