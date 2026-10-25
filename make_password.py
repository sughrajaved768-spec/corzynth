import streamlit_authenticator as stauth

passwords = [
    "Admin@2026",
    "Engineer@2026",
    "Viewer@2026"
]

try:
    # Newer versions of streamlit-authenticator
    hashed_passwords = [stauth.Hasher.hash(password) for password in passwords]
except AttributeError:
    # Older versions of streamlit-authenticator
    hashed_passwords = stauth.Hasher(passwords).generate()

for password in hashed_passwords:
    print(password)