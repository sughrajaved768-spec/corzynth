with open("app.py", encoding="utf-8") as f:
    lines = f.readlines()

count = 0
for i, line in enumerate(lines, 1):
    if '"""' in line:
        count += line.count('"""')
        print(f"Line {i}: {line.rstrip()}")

print(f"\nTotal triple-quote marks: {count}")
print("If this number is ODD, one \"\"\" is missing its partner.")