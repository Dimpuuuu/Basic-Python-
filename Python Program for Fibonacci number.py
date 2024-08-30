# Function to generate Fibonacci sequence up to n terms
def fibonacci(n):
    fib_sequence = []  # List to store Fibonacci numbers
    a, b = 0, 1        # Initial two terms of the sequence

    for _ in range(n):
        fib_sequence.append(a)  # Add the current number to the sequence
        a, b = b, a + b         # Update the values of a and b

    return fib_sequence

# Input: Number of terms to generate
n = int(input("Enter the number of terms: "))

# Output: Fibonacci sequence
print(f"Fibonacci sequence with {n} terms: {fibonacci(n)}")
