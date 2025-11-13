# --- Sample Data ---
users = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25},
    {'name': 'Charlie', 'age': 35}
]

# --- Manual Implementation ---
def manual_sort_by_key(data: list[dict], key: str) -> list[dict]:
    """
    Manually sorting a list of dictionaries by a key.
    The most efficient and Pythonic way is to use the built-in
    sorted() function with a lambda function as the key.
    This runs in O(n log n) time.
    """
    return sorted(data, key=lambda x: x[key])

# --- AI-Suggested Implementation ---
def ai_suggested_sort(data: list[dict], key: str) -> list[dict]:
    """
    Simulating a prompt to GitHub Copilot:
    "# python function to sort a list of dictionaries by key"
    
    The AI will almost certainly suggest the exact same optimal
    solution because it's trained on vast amounts of high-quality code.
    """
    return sorted(data, key=lambda x: x[key])

# --- Execution ---
if __name__ == "__main__":
    print("--- Task 1: AI Code Completion ---")
    
    print(f"Original list: {users}")
    
    manual_result = manual_sort_by_key(users, 'age')
    print(f"Manual sort by 'age': {manual_result}")
    
    ai_result = ai_suggested_sort(users, 'name')
    print(f"AI sort by 'name': {ai_result}")
