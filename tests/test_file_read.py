import os

print("🔥 TEST 1: File Read")

test_file = "sample.txt"

# create file
with open(test_file, "w") as f:
    f.write("Hello this is a test file")

# read file
with open(test_file, "r") as f:
    content = f.read()

print("✅ Content:", content)