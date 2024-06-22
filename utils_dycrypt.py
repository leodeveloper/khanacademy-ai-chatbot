# Decrypt the string
def decrypt_string(encrypted_base64, key):
    import base64
    from cryptography.fernet import Fernet

    fernet = Fernet(key)
    encrypted_bytes = base64.urlsafe_b64decode(encrypted_base64.encode())
    decrypted_string = fernet.decrypt(encrypted_bytes).decode()
    return decrypted_string