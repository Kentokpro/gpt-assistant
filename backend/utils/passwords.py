import random
import string

def generate_password(length=8):
    upper = random.choice(string.ascii_uppercase)
    others = [random.choice(string.ascii_lowercase + string.digits) for _ in range(length - 1)]
    others.insert(random.randint(0, length - 1), upper)
    return ''.join(others)
