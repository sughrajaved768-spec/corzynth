import os

password = "Admin@123"

try:
    import streamlit_authenticator as stauth
    try:
        hashed = stauth.Hasher.hash([password])[0]
    except AttributeError:
        hashed = stauth.Hasher([password]).generate()[0]
except Exception:
    import bcrypt
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

os.makedirs(".streamlit", exist_ok=True)

content = f"""[cookie]
name = "corzynth_auth_cookie"
key = "corzynth_super_secret_key_2026"
expiry_days = 30

[credentials.usernames.corzynth_admin]
name = "Corzynth Admin"
email = "admin@corzynth.com"
role = "admin"
password = "{hashed}"
"""

with open(".streamlit/secrets.toml", "w") as f:
    f.write(content)

print("SUCCESS!")
print("Username: corzynth_admin")
print("Password: Admin@123")