import numpy as np

arr0=[1,2,3]
dict={
    1: "0"
}

# 1. basic iteration over a list
for x in arr0:
    print(x)
# Output:
# 1
# 2
# 3

# 2. range-based loop
for i in range(5):
    print(i)
# Output:
# 0
# 1
# 2
# 3
# 4

# 3. range with start, stop, step
for i in range(2, 10, 2):
    print(i)
# Output:
# 2
# 4
# 6
# 8

# 4. enumerate — index + value
for idx, val in enumerate(arr0):
    print(idx, val)
# Output:
# 0 1
# 1 2
# 2 3

# 5. zip — iterate two lists in parallel
names = ["a", "b", "c"]
for name, num in zip(names, arr0):
    print(name, num)
# Output:
# a 1
# b 2
# c 3

# 6. iterate dict keys
for key in dict:
    print(key)
# Output:
# 1

# 7. iterate dict values
for value in dict.values():
    print(value)
# Output:
# 0

# 8. iterate dict items (key + value)
for key, value in dict.items():
    print(key, value)
# Output:
# 1 0

# 9. iterate over a string
for ch in "hello":
    print(ch)
# Output:
# h
# e
# l
# l
# o

# 10. reversed
for x in reversed(arr0):
    print(x)
# Output:
# 3
# 2
# 1

# 11. sorted
for x in sorted(arr0, reverse=True):
    print(x)
# Output:
# 3
# 2
# 1

# 12. nested loop
for i in arr0:
    for j in names:
        print(i, j)
# Output:
# 1 a
# 1 b
# 1 c
# 2 a
# 2 b
# 2 c
# 3 a
# 3 b
# 3 c

# 13. for-else (else runs if loop completes without break)
for x in arr0:
    if x == 99:
        break
else:
    print("not found")
# Output:
# not found

# 14. list comprehension
squares = [x * x for x in arr0]
print(squares)
# Output:
# [1, 4, 9]

# 15. dict comprehension
sq_map = {x: x * x for x in arr0}
print(sq_map)
# Output:
# {1: 1, 2: 4, 3: 9}

# 16. set comprehension
unique = {x % 2 for x in arr0}
print(unique)
# Output:
# {0, 1}

# 17. generator expression (lazy)
gen = (x * 2 for x in arr0)
for v in gen:
    print(v)
# Output:
# 2
# 4
# 6

# 18. iterate numpy array (mixed types coerced to strings)
arr = np.array(["hi", 1, 2])
for item in arr:
    print(item)
# Output:
# hi
# 1
# 2

# 19. while-style with iter() + next()
it = iter(arr0)
while True:
    try:
        print(next(it))
    except StopIteration:
        break
# Output:
# 1
# 2
# 3

# ─── JS-style loops (start, stop, step + continue/break) ───

# JS:  for (let i = 0; i < 10; i++)
for i in range(0, 10, 1):
    print(i)
# Output:
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9

# JS:  for (let i = 5; i < 20; i += 3)
for i in range(5, 20, 3):
    print(i)
# Output:
# 5
# 8
# 11
# 14
# 17

# JS:  for (let i = 10; i > 0; i--)   — decrement with negative step
for i in range(10, 0, -1):
    print(i)
# Output:
# 10
# 9
# 8
# 7
# 6
# 5
# 4
# 3
# 2
# 1

# JS:  for (let i = 0; i < 10; i++) { if (i % 2) continue; ... }
for i in range(0, 10):
    if i % 2 != 0:
        continue              # skip odd numbers
    print("even:", i)
# Output:
# even: 0
# even: 2
# even: 4
# even: 6
# even: 8

# JS:  for (...) { if (cond) break; }
for i in range(0, 100):
    if i > 5:
        break                 # stop early
    print("under 6:", i)
# Output:
# under 6: 0
# under 6: 1
# under 6: 2
# under 6: 3
# under 6: 4
# under 6: 5

# JS:  combined continue + break
for i in range(0, 20):
    if i == 3:
        continue              # skip 3
    if i == 10:
        break                 # stop at 10
    print("value:", i)
# Output:
# value: 0
# value: 1
# value: 2
# value: 4
# value: 5
# value: 6
# value: 7
# value: 8
# value: 9

# Float step (range only takes ints) — use numpy.arange
for i in np.arange(0.0, 1.0, 0.1):
    print(round(i, 2))
# Output:
# 0.0
# 0.1
# 0.2
# 0.3
# 0.4
# 0.5
# 0.6
# 0.7
# 0.8
# 0.9

# Pure while-loop version (most JS-like)
i = 0
while i < 10:
    if i == 4:
        i += 1
        continue              # must increment before continue!
    if i == 8:
        break
    print("while:", i)
    i += 1
# Output:
# while: 0
# while: 1
# while: 2
# while: 3
# while: 5
# while: 6
# while: 7

print("hello world", arr0, arr, dict)
# Output:
# hello world [1, 2, 3] ['hi' '1' '2'] {1: '0'}
