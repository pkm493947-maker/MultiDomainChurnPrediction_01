def generate_retention_strategy(metrics):

    print("\n💡 Generating Retention Strategy...")

    if metrics["recall"] < 0.70:
        print("⚠ Improve recall using class balancing or more data.")
    else:
        print("✅ Recall is good. Focus on targeted retention campaigns.")