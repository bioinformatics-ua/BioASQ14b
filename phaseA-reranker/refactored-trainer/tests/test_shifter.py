"""
Test script to verify ShifterSampler is now compatible with BasicSampler interface.
Demonstrates the curriculum learning behavior across epochs.
"""

import sys
sys.path.insert(0, '/home/ucloud/BioASQ13B/phaseA-reranker/refactored-trainer')

from sampler import BasicSampler, ShifterSampler


def create_mock_dataset():
    """Create mock dataset with many negatives to show the shifting effect."""
    return {
        "query_001": {
            "question": "What is diabetes treatment?",
            0: [{"id": f"neg_{i:03d}", "text": f"Negative document {i}"} for i in range(100)],
            1: [{"id": "pos_001", "text": "Insulin therapy is the treatment for diabetes."}],
        }
    }


def main():
    print("=" * 80)
    print("SHIFTERSAMPLER COMPATIBILITY TEST")
    print("=" * 80)

    dataset = create_mock_dataset()
    max_epoch = 5  # Simulate 5 epochs

    print("\n1. Testing BasicSampler (control - should work as before):")
    print("-" * 80)

    basic_sampler = BasicSampler(dataset, collection=None)
    print("   Created BasicSampler successfully")
    print(f"   - relevance_order: {basic_sampler.relevance_order}")
    print(f"   - negative_index: {basic_sampler.negative_index}")
    print(f"   - positive_index: {basic_sampler.positive_index}")

    # Sample across epochs (BasicSampler ignores epoch)
    for epoch in range(3):
        q_id, q_text = basic_sampler.choose_question(0, epoch)
        neg = basic_sampler.choose_negative_doc(0, epoch, q_id)
        if neg:
            print(f"   Epoch {epoch}: sampled neg ID starting with '{neg[:15]}...'")
        else:
            print(f"   Epoch {epoch}: no negative sampled")

    print("\n2. Testing ShifterSampler (new - same interface):")
    print("-" * 80)

    shifter_sampler = ShifterSampler(dataset, collection=None, max_epoch=max_epoch)
    print("   Created ShifterSampler successfully")
    print(f"   - max_epoch: {shifter_sampler.max_epoch}")
    print(f"   - relevance_order: {shifter_sampler.relevance_order}")
    print(f"   - negative_index: {shifter_sampler.negative_index}")
    print(f"   - positive_index: {shifter_sampler.positive_index}")

    print(f"\n   Sampling negatives across {max_epoch} epochs:")
    print(f"   (With 100 negatives and max_epoch=5, interval = {100 // (max_epoch + 1)} = 16)")
    print()

    for epoch in range(max_epoch + 1):
        q_id, q_text = shifter_sampler.choose_question(0, epoch)

        # Sample multiple times to show the range
        samples = []
        for _ in range(5):
            neg = shifter_sampler.choose_negative_doc(0, epoch, q_id)
            # Extract the number from "Negative document X"
            if neg:
                try:
                    num = int(neg.split()[-1])
                    samples.append(num)
                except (ValueError, IndexError):
                    samples.append(-1)
            else:
                samples.append(-1)

        interval = 100 // (max_epoch + 1)
        start_pos = interval * epoch
        print(f"   Epoch {epoch}: samples from negatives [{start_pos}:99]")
        print(f"            sampled values: {samples} (min: {min(samples)}, max: {max(samples)})")

    print("\n3. Verifying method signatures match:")
    print("-" * 80)
    
    methods_to_check = [
        "__init__",
        "choose_question", 
        "choose_positive_doc",
        "choose_negative_doc",
    ]
    
    import inspect
    
    for method_name in methods_to_check:
        basic_sig = inspect.signature(getattr(BasicSampler, method_name))
        shifter_sig = inspect.signature(getattr(ShifterSampler, method_name))
        
        # Check parameter names match (ignoring self)
        basic_params = list(basic_sig.parameters.keys())
        shifter_params = list(shifter_sig.parameters.keys())
        
        match = basic_params == shifter_params
        status = "✓" if match else "✗"
        print(f"   {status} {method_name}: {shifter_params}")

    print("\n4. Summary:")
    print("-" * 80)
    print("   ShifterSampler now:")
    print("   - Inherits from BasicSampler (same interface)")
    print("   - Accepts 'collection' parameter like parent")
    print("   - Uses 'max_epoch' from kwargs")
    print("   - Uses standard keys (0, 1) from relevance_order")
    print("   - Implements curriculum learning via epoch parameter")
    print("=" * 80)


if __name__ == "__main__":
    main()
