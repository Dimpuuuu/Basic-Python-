# Define two sets
set_A = {1, 2, 3, 4, 5}
set_B = {4, 5, 6, 7, 8}

# Display the sets
print(f"Set A: {set_A}")
print(f"Set B: {set_B}")

# Union of sets
union_result = set_A | set_B
print(f"Union of A and B: {union_result}")

# Intersection of sets
intersection_result = set_A & set_B
print(f"Intersection of A and B: {intersection_result}")

# Difference of sets (A - B)
difference_result = set_A - set_B
print(f"Difference of A and B (A - B): {difference_result}")

# Difference of sets (B - A)
difference_result_B_A = set_B - set_A
print(f"Difference of B and A (B - A): {difference_result_B_A}")

# Symmetric difference of sets
symmetric_difference_result = set_A ^ set_B
print(f"Symmetric Difference of A and B: {symmetric_difference_result}")
