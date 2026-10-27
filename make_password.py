import os

password1 = "Corzynth@Secure2026!"
password2 = "JanAlam@2026"

try:
    import streamlit_authenticator as stauth
    try:
        hashed1 = stauth.Hasher.hash([password1])[0]
        hashed2 = stauth.Hasher.hash([password2])[0]
    except AttributeError:
        hashed1 = stauth.Hasher([password1]).generate()[0]
        hashed2 = stauth.Hasher([password2]).generate()[0]
except Exception:
    import bcrypt
    hashed1 = bcrypt.hashpw(password1.encode(), bcrypt.gensalt()).decode()
    hashed2 = bcrypt.hashpw(password2.encode(), bcrypt.gensalt()).decode()

os.makedirs(".streamlit", exist_ok=True)

content = f"""[cookie]
name = "corzynth_auth_cookie"
key = "Nq8Wm2Rv7Xz4Kp9Ld3Jf6Hs1Tb5Yc0AeWq8Nc2Vb5Tm1Zr4Yh6Ju3Pd0Ks7Xf92"
expiry_days = 30

[credentials.usernames.corzynth_admin]
name = "Corzynth Admin"
email = "admin@corzynth.com"
role = "admin"
password = "{hashed1}"

[credentials.usernames.jan_alam]
name = "Jan Alam"
email = "janalam@corzynth.com"
role = "admin"
password = "{hashed2}"
"""

with open(".streamlit/secrets.toml", "w") as f:
    f.write(content)

print("SUCCESS! Two admins created:")
print("  1) Username: corzynth_admin  | Password: Corzynth@Secure2026!")
print("  2) Username: jan_alam        | Password: JanAlam@2026")