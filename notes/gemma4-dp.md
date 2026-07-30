# Gemma 4 E4B Test

The original Levenshtein Distance implementation was a straight-forward translation
of the recursive mathematical definition, which resulted in RecursionError for
sufficiently large strings. The solution was to use dynamic programming to convert
it from a recursive function to a function that explored a state space in as tight
a loop as possible.

Since I am curious to see the extent of the capabilities of small, local models, I
decided to give Gemma 4 E4B a shot. In the first message, I asked it to explain the
concept of dynamic programming and work through an example using a Fibonacci number
generator. On the second turn, I gave it the recursive `levenshtein_distance`
function and told it to rewrite it with dynamic programming. I then copied and
pasted the output to test, and it failed, returning 0 for all strings. I gave it one
more chance and told it the problem, then copied and pasted what it output. It had
changed the API by dropping the `normalize` argument and creating a separate
`levenshtein_distance_normalized` function, making it incompatible with the test
suite, so I added that argument and a ternary to the return statement and re-ran the
test suite as well as some additional comparison fuzzers. To my surprise, it worked.

I did not read through the generated code because it greatly offended my aesthetic
senses, and I did not want it planting any dumb ideas in my mind about how to solve
this. I decided to try my hand at it in the morning after sleep and coffee, and I
committed the result of that effort to the repo. I then decided to compare my
first solution to the clanker's solution, which I have preserved below.

```python
def levenshtein_distance(a: str, b: str) -> int:
    """
    Calculates Levenshtein distance using dynamic programming (tabulation).
    Time Complexity: O(len(a) * len(b))
    Space Complexity: O(len(a) * len(b))
    """
    m = len(a)
    n = len(b)

    # Optimization for Space: We only need the previous row (i-1) and the current row (i) to calculate the distance.
    # This reduces space complexity from O(m*n) to O(min(m, n)).
    if m < n:
        # Ensure 'a' is always the longer string for consistent array sizing
        return levenshtein_distance(b, a)

    # Initialize two arrays: prev_row (i-1) and current_row (i)
    prev_row = list(range(n + 1)) # Base case for row 0: [0, 1, 2, ..., n]
    current_row = [0] * (n + 1)

    # Iterate through the characters of 'a' (the outer loop, simulating rows i=1 to m)
    for i in range(1, m + 1):
        # The first column entry for the current row must be initialized to 'i' (deletion cost)
        current_row[0] = i 

        # Iterate through the characters of 'b' (the inner loop, simulating columns j=1 to n)
        for j in range(1, n + 1):
            
            # Calculate substitution cost: 0 if chars match, 1 otherwise.
            cost = 0 if a[i - 1] == b[j - 1] else 1

            # Apply the Levenshtein recurrence relation (the core logic)
            current_row[j] = min(
                prev_row[j] + 1, # Deletion cost (from above: dp[i-1][j])
                current_row[j - 1] + 1, # Insertion cost (from left: dp[i][j-1])
                prev_row[j - 1] + cost # Substitution/Match cost (from diagonal: dp[i-1][j-1])
            )

        # After completing row 'i', the current_row becomes the previous row for the next iteration.
        prev_row = list(current_row)
        # Reset the current_row to prepare for the next iteration
        current_row = [0] * (n + 1)

    # The result is stored in the last computed value of the final row
    return prev_row[n]

def levenshtein_distance_normalized(a: str, b: str) -> float:
    return levenshtein_distance(a, b) / max(len(a), len(b))
```

There is a lot that is dumb and ugly about this code, but it did encode a more
space-efficient solution than my first attempt. I did a second optimization pass for
my implementation and validated that it used an order of magnitude less memory. The
result in the library is my hand-written code.
